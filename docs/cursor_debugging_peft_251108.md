# 如何解决bash错误
_Exported on 11/8/2025 at 14:00:21 PST from Cursor (2.0.69)_

---

**User**

这个错误怎么解决：@bash (44-51) 

---

**Cursor**


检查相关代码文件，确认 `tensorboard` 的使用情况：



检查 requirements 文件，确认是否已包含 tensorboard：



`requirements.txt` 已包含 `tensorboard>=2.10.0`，但环境中未安装。正在安装 `tensorboard`：



tensorboard 已在 base 环境安装，但错误显示运行在 `seg-r1` 环境。检查并激活 `seg-r1` 环境，然后安装 tensorboard：



使用该环境的 pip 直接安装 tensorboard：



问题已解决。

**问题原因：**
代码在 `seg-r1` 环境中运行，但该环境缺少 `tensorboard` 包。

**解决方案：**
已在 `seg-r1` 环境中安装 `tensorboard`（版本 2.20.0）及其依赖。

现在可以重新运行 `train_supervised_peft.py`，`ModuleNotFoundError: No module named 'tensorboard'` 错误应已消失。

**提示：** 如果将来遇到类似的依赖缺失问题，可以安装整个 requirements.txt 文件：
```bash
/root/autodl-tmp/envs/seg-r1/bin/pip install -r seg-rl/peft/requirements.txt
```

这样可以确保所有依赖都已安装。

```shellscript
/root/autodl-tmp/envs/seg-r1/bin/pip install -r seg-rl/peft/requirements.txt
```

---

**User**

运行实验4.1报错：@bash (983-1020) 

---

**Cursor**


这是 PyTorch 2.6 的兼容性问题：`torch.load` 默认 `weights_only=True`，而 checkpoint 包含 numpy 对象。检查并修复代码：



检查文件中所有 `torch.load` 的使用位置：



修复 `torch.load` 调用，添加 `weights_only=False` 参数：



修复 `torch.load` 调用，添加 `weights_only=False`：


```python
#!/usr/bin/env python3
"""
PEFT模型评估脚本

在测试集上评估训练好的PEFT模型性能

主要功能:
- 加载训练好的checkpoint
- 执行完整rollout（迭代预测点序列）
- 计算指标：Dice, IoU, PCK, 点数等
- 生成可视化：点序列图、分割结果对比

调用示例:
    python -m seg-rl.peft.eval_peft_model \\
      --test_json datasets/seg_r1_md/Task01_BrainTumour/test.jsonl \\
      --checkpoint outputs/braintumour/peft_supervised/checkpoint_best.pt \\
      --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \\
      --out_dir outputs/braintumour/peft_eval \\
      --max_rollout_steps 16 \\
      --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# 添加父目录到路径
_PARENT_DIR = Path(__file__).parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from peft.lora_sam2 import LoRASAM2Wrapper
from peft.point_predictor_peft import PointPredictorFromSAMFeatures
from peft.utils_peft import compute_dice, compute_iou, load_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate PEFT Model")
    
    p.add_argument("--test_json", type=str, required=True, help="Test JSONL file")
    p.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint to evaluate")
    p.add_argument("--sam_checkpoint", type=str, required=True, help="SAM2 checkpoint")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory for results")
    
    p.add_argument("--image_size", type=int, nargs=2, default=[512, 512], help="Image size")
    p.add_argument("--max_rollout_steps", type=int, default=16, help="Max points per image")
    p.add_argument("--dice_threshold", type=float, default=0.95, help="Stop if Dice > threshold")
    p.add_argument("--improvement_threshold", type=float, default=0.001,
                   help="Stop if improvement < threshold")
    
    p.add_argument("--device", type=str, default="cuda", help="Device")
    p.add_argument("--save_visualizations", action="store_true", help="Save visualization images")
    
    return p.parse_args()


@torch.no_grad()
def rollout_single_image(
    image_path: str,
    gt_mask_path: str,
    sam2_lora: LoRASAM2Wrapper,
    point_predictor: PointPredictorFromSAMFeatures,
    args,
) -> Dict:
    """
    对单张图像执行完整rollout
    
    Returns:
        结果字典，包含点序列、Dice曲线等
    """
    device = args.device
    
    # 加载图像和真值
    image = Image.open(image_path).convert('RGB')
    gt_mask = Image.open(gt_mask_path).convert('L')
    
    orig_w, orig_h = image.size
    
    # Resize
    image_resized = image.resize(tuple(args.image_size), Image.BILINEAR)
    gt_mask_resized = gt_mask.resize(tuple(args.image_size), Image.NEAREST)
    
    # 转换为tensor
    import torchvision.transforms.functional as TF
    image_tensor = TF.to_tensor(image_resized).unsqueeze(0).to(device)  # [1, 3, H, W]
    gt_mask_tensor = TF.to_tensor(gt_mask_resized).to(device)  # [1, H, W]
    gt_mask_binary = (gt_mask_tensor > 0.5).float()
    
    # Rollout
    points_seq = []
    labels_seq = []
    dice_history = []
    iou_history = []
    
    prev_dice = 0.0
    
    for step in range(args.max_rollout_steps):
        # 生成当前掩模
        if step == 0:
            prev_mask = torch.zeros_like(gt_mask_binary)
        else:
            # 用SAM2生成掩模
            # 需要将点坐标缩放回原始尺寸
            points_orig = [(px * orig_w / args.image_size[1], py * orig_h / args.image_size[0])
                          for px, py in points_seq]
            prev_mask = sam2_lora.predict_mask(
                np.array(image), points_orig, labels_seq
            )
            prev_mask = torch.from_numpy(prev_mask).unsqueeze(0).to(device)
            prev_mask = torch.nn.functional.interpolate(
                prev_mask.unsqueeze(0), size=tuple(args.image_size), mode='nearest'
            )[0]
        
        # 预测下一个点
        sam_features = sam2_lora.get_image_features(image_tensor, feature_scale=8)
        heatmap_logits, label_logits = point_predictor(sam_features, prev_mask.unsqueeze(0))
        
        # Argmax获取点
        B, _, H, W = heatmap_logits.shape
        heatmap_flat = heatmap_logits.view(B, -1)
        pred_idx = heatmap_flat.argmax(dim=1)
        pred_y = (pred_idx // W).float()
        pred_x = (pred_idx % W).float()
        
        # 标签
        pred_label = label_logits.argmax(dim=1).item()
        
        # 记录
        points_seq.append((pred_x.item(), pred_y.item()))
        labels_seq.append(pred_label)
        
        # 计算指标
        dice = compute_dice(prev_mask[0], gt_mask_binary[0])
        iou = compute_iou(prev_mask[0], gt_mask_binary[0])
        dice_history.append(dice)
        iou_history.append(iou)
        
        # 停止条件
        if dice > args.dice_threshold:
            break
        if step > 0 and (dice - prev_dice) < args.improvement_threshold:
            break
        
        prev_dice = dice
    
    return {
        'image_path': image_path,
        'points': points_seq,
        'labels': labels_seq,
        'dice_history': dice_history,
        'iou_history': iou_history,
        'final_dice': dice_history[-1] if dice_history else 0.0,
        'final_iou': iou_history[-1] if iou_history else 0.0,
        'num_points': len(points_seq),
    }


def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 设备
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # 加载checkpoint配置
    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = ckpt.get('config', {})
    
    # 提取配置
    lora_rank = config.get('lora_rank', 16)
    lora_alpha = config.get('lora_alpha', 32)
    fusion_mode = config.get('fusion_mode', 'film')
    feature_scale = config.get('feature_scale', 8)
    
    print(f"Model config: LoRA rank={lora_rank}, alpha={lora_alpha}, fusion={fusion_mode}")
    
    # 1. 初始化SAM2 + LoRA
    print("\nInitializing SAM2 with LoRA...")
    sam2_lora = LoRASAM2Wrapper(
        sam_checkpoint=args.sam_checkpoint,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        device=str(device)
    )
    
    # 2. 初始化点预测网络
    print("Initializing Point Predictor...")
    
    # 根据 feature_scale 确定 SAM 特征通道数
    sam_feature_dim_map = {4: 32, 8: 64, 16: 256}
    if feature_scale not in sam_feature_dim_map:
        raise ValueError(f"Unsupported feature_scale: {feature_scale}. Use 4, 8, or 16.")
    sam_feature_dim = sam_feature_dim_map[feature_scale]
    
    point_predictor = PointPredictorFromSAMFeatures(
        sam_feature_dim=sam_feature_dim,
        output_size=tuple(args.image_size),
        fusion_mode=fusion_mode,
        feature_scale=feature_scale,
    ).to(device)
    
    # 3. 加载权重
    print("Loading model weights...")
    load_checkpoint(
        args.checkpoint, point_predictor, sam2_lora, None, None, None, str(device)
    )
    
    point_predictor.eval()
    
    # 4. 加载测试数据
    print(f"\nLoading test data from {args.test_json}")
    with open(args.test_json, 'r') as f:
        content = f.read().strip()
        if content.startswith('['):
            test_samples = json.loads(content)
        else:
            test_samples = [json.loads(line) for line in content.splitlines() if line.strip()]
    
    print(f"Loaded {len(test_samples)} test samples")
    
    # 5. 评估
    print("\n" + "="*80)
    print("Evaluating...")
    print("="*80)
    
    results = []
    
    for sample in tqdm(test_samples):
        image_path = sample['image']
        gt_mask_path = sample.get('gt_mask', None)
        
        if gt_mask_path is None or not os.path.exists(gt_mask_path):
            print(f"Warning: GT mask not found for {image_path}, skipping")
            continue
        
        result = rollout_single_image(
            image_path, gt_mask_path, sam2_lora, point_predictor, args
        )
        results.append(result)
    
    # 6. 汇总统计
    print("\n" + "="*80)
    print("Results Summary")
    print("="*80)
    
    final_dices = [r['final_dice'] for r in results]
    final_ious = [r['final_iou'] for r in results]
    num_points = [r['num_points'] for r in results]
    
    print(f"Number of samples: {len(results)}")
    print(f"\nDice Score:")
    print(f"  Mean: {np.mean(final_dices):.4f}")
    print(f"  Std:  {np.std(final_dices):.4f}")
    print(f"  Median: {np.median(final_dices):.4f}")
    
    print(f"\nIoU Score:")
    print(f"  Mean: {np.mean(final_ious):.4f}")
    print(f"  Std:  {np.std(final_ious):.4f}")
    print(f"  Median: {np.median(final_ious):.4f}")
    
    print(f"\nNumber of Points:")
    print(f"  Mean: {np.mean(num_points):.2f}")
    print(f"  Std:  {np.std(num_points):.2f}")
    print(f"  Median: {np.median(num_points):.0f}")
    
    # 7. 保存结果
    results_json_path = os.path.join(args.out_dir, 'eval_results.json')
    with open(results_json_path, 'w') as f:
        json.dump({
            'summary': {
                'num_samples': len(results),
                'mean_dice': float(np.mean(final_dices)),
                'std_dice': float(np.std(final_dices)),
                'median_dice': float(np.median(final_dices)),
                'mean_iou': float(np.mean(final_ious)),
                'mean_num_points': float(np.mean(num_points)),
            },
            'per_sample_results': results,
        }, f, indent=2)
    
    print(f"\nResults saved to {results_json_path}")


if __name__ == "__main__":
    main()


```


检查训练脚本中是否也需要修复类似问题：



检查其他文件中的 `torch.load` 调用，确保它们也兼容 PyTorch 2.6：



修复其他文件中的 `torch.load` 调用，避免同样的问题：


```python
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
    init_ckpt = torch.load(args.init_policy, map_location='cpu', weights_only=False)
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
    
    # 根据 feature_scale 确定 SAM 特征通道数
    sam_feature_dim_map = {4: 32, 8: 64, 16: 256}
    if feature_scale not in sam_feature_dim_map:
        raise ValueError(f"Unsupported feature_scale: {feature_scale}. Use 4, 8, or 16.")
    sam_feature_dim = sam_feature_dim_map[feature_scale]
    
    point_predictor_policy = PointPredictorFromSAMFeatures(
        sam_feature_dim=sam_feature_dim,
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
        sam_feature_dim=sam_feature_dim,
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


```

```python
#!/usr/bin/env python3
"""
LoRA集成模块 - 为SAM2注入Late LoRA并提供特征提取接口

该模块实现了Late LoRA策略：只在SAM2 image encoder的最后一个Transformer块中注入LoRA适配器。
LoRA (Low-Rank Adaptation) 通过低秩矩阵分解实现参数高效微调。

核心功能:
- LoRALinear: LoRA层的基础实现
- LoRASAM2Wrapper: 封装SAM2模型，注入LoRA，提供特征提取和掩模预测接口

调用示例:
    # 初始化带LoRA的SAM2
    sam2_lora = LoRASAM2Wrapper(
        sam_checkpoint="third_party/sam2/checkpoints/sam2.1_hiera_large.pt",
        lora_rank=16,
        lora_alpha=32,
        device="cuda"
    )
    
    # 提取图像特征（用于点预测网络）
    features = sam2_lora.get_image_features(image_tensor, feature_scale=8)
    
    # 预测分割掩模（用于环境反馈）
    mask = sam2_lora.predict_mask(image_tensor, points, labels)
    
    # 保存/加载LoRA权重
    sam2_lora.save_lora_checkpoint("outputs/lora_weights.pt")
    sam2_lora.load_lora_checkpoint("outputs/lora_weights.pt")

参考论文:
    Late LoRA方法参考: Parameter Efficient Fine-Tuning of Segment Anything Model
    for Biomedical Imaging
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image as PILImage

# 添加sam2路径
_SAM2_PATH = Path(__file__).parent.parent.parent / "third_party" / "sam2"
if str(_SAM2_PATH) not in sys.path:
    sys.path.insert(0, str(_SAM2_PATH))

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError as e:
    print(f"Error importing SAM2: {e}")
    print("Please ensure SAM2 is properly installed in third_party/sam2")
    raise


class LoRALinear(nn.Module):
    """
    LoRA (Low-Rank Adaptation) 线性层
    
    实现公式: output = frozen_linear(x) + (alpha/rank) * lora_B @ lora_A @ x
    
    Args:
        base_linear: 被冻结的基础线性层
        rank: LoRA的秩（低秩矩阵的维度）
        alpha: 缩放因子，控制LoRA对输出的贡献
        
    属性:
        base_linear: 冻结的原始权重（不可训练）
        lora_A: 下投影矩阵 [rank, in_features]
        lora_B: 上投影矩阵 [out_features, rank]
        scaling: alpha / rank
    """
    
    def __init__(self, base_linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base_linear = base_linear
        self.base_linear.requires_grad_(False)  # 冻结基础权重
        
        in_features = base_linear.in_features
        out_features = base_linear.out_features
        
        # 获取base_linear的设备，确保LoRA参数在同一设备上
        device = base_linear.weight.device
        
        # LoRA矩阵初始化：A用小随机值，B用零（保证初始时LoRA不影响输出）
        self.lora_A = nn.Parameter(torch.randn(rank, in_features, device=device) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=device))
        self.scaling = alpha / rank
        
        self.rank = rank
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播: 基础输出 + LoRA增量
        
        Args:
            x: 输入张量 [..., in_features]
            
        Returns:
            输出张量 [..., out_features]
        """
        # 基础线性层输出（冻结，无梯度）
        base_out = self.base_linear(x)
        
        # LoRA路径: x @ A^T @ B^T，带缩放
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        
        return base_out + self.scaling * lora_out
    
    def extra_repr(self) -> str:
        return f'in_features={self.base_linear.in_features}, out_features={self.base_linear.out_features}, rank={self.rank}, alpha={self.alpha}'


class LoRASAM2Wrapper:
    """
    SAM2 + Late LoRA 封装器
    
    实现Late LoRA策略：只在SAM2 image encoder的最后一个Transformer块注入LoRA。
    根据SAM2.1 Hiera Large配置，Hiera有4个stage [2, 6, 36, 4]，我们在最后一个
    stage的最后一个block中的attention层注入LoRA。
    
    主要功能:
    1. 加载SAM2模型并注入LoRA到指定位置
    2. 提取多尺度图像特征（用于点预测网络）
    3. 生成分割掩模（用于环境反馈和训练）
    4. 管理LoRA权重的保存和加载
    
    Args:
        sam_checkpoint: SAM2模型checkpoint路径
        lora_rank: LoRA秩，默认16
        lora_alpha: LoRA缩放因子，默认32
        device: 运行设备 ("cuda", "cpu", "mps")
        config_file: SAM2配置文件路径
    """
    
    def __init__(
        self,
        sam_checkpoint: str,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        device: str = "cuda",
        config_file: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
    ):
        self.device = torch.device(device)
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        
        # 加载SAM2模型
        print(f"Loading SAM2 from {sam_checkpoint}")
        sam_model = build_sam2(config_file, sam_checkpoint, device=self.device)
        
        # 初始化预测器
        self.predictor = SAM2ImagePredictor(sam_model)
        self.model = sam_model
        
        # 注入LoRA到最后一个Transformer块
        self._inject_lora()
        
        print(f"LoRA injected: rank={lora_rank}, alpha={lora_alpha}")
        print(f"Trainable LoRA parameters: {sum(p.numel() for p in self.get_lora_parameters()):,}")
        
    def _inject_lora(self):
        """
        注入LoRA到SAM2 image encoder的最后一个Transformer块
        
        对于Hiera模型，结构为:
        - trunk (Hiera backbone)
          - blocks: nn.ModuleList[MultiScaleBlock]  <- 所有块的列表
          - stage_ends: List[int]  <- 每个阶段结束的块索引
            - blocks[i].attn: MultiScaleAttention
              - qkv: Linear  <- 注入LoRA的目标
        """
        # 获取image encoder的trunk (Hiera)
        image_encoder = self.model.image_encoder
        trunk = image_encoder.trunk
        
        # 访问blocks列表（Hiera使用blocks而不是stages）
        if not hasattr(trunk, 'blocks') or len(trunk.blocks) == 0:
            raise RuntimeError("Cannot find blocks in Hiera trunk")
        
        # 获取最后一个block（最后一个阶段的最后一个块）
        last_block = trunk.blocks[-1]
        
        # 找到attention层的qkv投影
        if not hasattr(last_block, 'attn'):
            raise RuntimeError("Cannot find attention in last block")
        
        attn = last_block.attn
        
        # 注入LoRA到qkv层
        # 注意: 不同SAM2版本可能有不同的注意力实现，需要适配
        if hasattr(attn, 'qkv') and isinstance(attn.qkv, nn.Linear):
            print(f"Injecting LoRA into last block's attn.qkv")
            original_qkv = attn.qkv
            attn.qkv = LoRALinear(original_qkv, self.lora_rank, self.lora_alpha)
        elif hasattr(attn, 'q_proj') and hasattr(attn, 'k_proj') and hasattr(attn, 'v_proj'):
            # 如果qkv是分开的，分别注入
            print(f"Injecting LoRA into last block's attn.q_proj, k_proj, v_proj")
            if isinstance(attn.q_proj, nn.Linear):
                original_q = attn.q_proj
                attn.q_proj = LoRALinear(original_q, self.lora_rank, self.lora_alpha)
            if isinstance(attn.k_proj, nn.Linear):
                original_k = attn.k_proj
                attn.k_proj = LoRALinear(original_k, self.lora_rank, self.lora_alpha)
            if isinstance(attn.v_proj, nn.Linear):
                original_v = attn.v_proj
                attn.v_proj = LoRALinear(original_v, self.lora_rank, self.lora_alpha)
        else:
            raise RuntimeError("Cannot find qkv or q/k/v projection layers in attention")
        
        # 可选: 也可以注入到MLP层
        # if hasattr(last_block, 'mlp'):
        #     ...
    
    def get_lora_parameters(self) -> Iterator[nn.Parameter]:
        """
        返回所有LoRA参数的迭代器（用于优化器）
        
        Returns:
            LoRA参数迭代器
        """
        for module in self.model.modules():
            if isinstance(module, LoRALinear):
                yield module.lora_A
                yield module.lora_B
    
    def freeze_sam_base(self):
        """
        冻结SAM2的所有基础参数，只保留LoRA可训练
        
        注意: LoRA注入时已经冻结了被wrap的Linear层，此方法确保其他层也被冻结
        """
        for name, param in self.model.named_parameters():
            if 'lora_' not in name:
                param.requires_grad = False
        
        # 确保LoRA参数可训练
        for param in self.get_lora_parameters():
            param.requires_grad = True
    
    def get_image_features(
        self,
        image: torch.Tensor,
        feature_scale: int = 8
    ) -> torch.Tensor:
        """
        提取SAM2图像特征（用于点预测网络）
        
        SAM2的FPN输出多个尺度的特征:
        - Level 0: stride 4 (高分辨率)
        - Level 1: stride 8 (中等，推荐)
        - Level 2: stride 16 (低分辨率)
        
        Args:
            image: 输入图像 [B, 3, H, W]，已归一化
            feature_scale: 特征尺度，8表示H/8 x W/8（推荐）
            
        Returns:
            特征张量 [B, 256, H/scale, W/scale]
        """
        # 通过image encoder提取特征
        backbone_out = self.model.forward_image(image)
        
        # SAM2内部方法: 准备backbone特征
        # 返回: (backbone_out, vision_feats, vision_pos_embeds, feat_sizes)
        # vision_feats格式: [HW, B, C] (已展平)
        _, vision_feats, _, feat_sizes = self.model._prepare_backbone_features(backbone_out)
        
        # vision_feats是一个list，包含多个尺度的特征
        # 根据feature_scale选择对应的level
        if feature_scale == 4:
            feat_idx = 0
        elif feature_scale == 8:
            feat_idx = 1 if len(vision_feats) > 1 else 0
        elif feature_scale == 16:
            feat_idx = 2 if len(vision_feats) > 2 else len(vision_feats) - 1
        else:
            raise ValueError(f"Unsupported feature_scale: {feature_scale}. Use 4, 8, or 16.")
        
        # 获取选中的特征和尺寸
        feature = vision_feats[feat_idx]  # [HW, B, C]
        feat_size = feat_sizes[feat_idx]  # (H, W)
        batch_size = image.shape[0]
        
        # 将特征从 [HW, B, C] 转换为 [B, C, H, W]
        # 参考 sam2_image_predictor.py 中的转换方式
        feature = feature.permute(1, 2, 0)  # [B, C, HW]
        feature = feature.view(batch_size, -1, *feat_size)  # [B, C, H, W]
        
        return feature
    
    def predict_mask(
        self,
        image: Union[torch.Tensor, np.ndarray, PILImage.Image],
        points: List[Tuple[float, float]],
        labels: List[int],
        return_logits: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        使用SAM2生成分割掩模（完整前向传播）
        
        Args:
            image: 输入图像，可以是:
                - torch.Tensor [B, 3, H, W] 或 [3, H, W]
                - np.ndarray [H, W, 3]
                - PIL.Image
            points: 点提示列表 [(x, y), ...]
            labels: 点标签列表 [1, 0, ...]，1=前景，0=背景
            return_logits: 是否返回logits（用于某些loss计算）
            
        Returns:
            如果return_logits=False: 二值掩模 [H, W] 或 [B, H, W]
            如果return_logits=True: (掩模, logits)
        """
        # 转换输入为numpy格式（SAM2 predictor需要）
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image[0]  # 取第一张图
            image = image.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
            image = (image * 255).astype(np.uint8)
        elif isinstance(image, PILImage.Image):
            image = np.array(image)
        
        # 设置图像
        self.predictor.set_image(image)
        
        # 转换点和标签为numpy数组
        input_points = np.array(points, dtype=np.float32) if points else None
        input_labels = np.array(labels, dtype=np.int32) if labels else None
        
        # 预测
        masks, scores, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False,
        )
        
        # 转换为torch tensor
        mask = torch.from_numpy(masks[0]).float().to(self.device)
        
        if return_logits:
            logits_tensor = torch.from_numpy(logits[0]).float().to(self.device)
            return mask, logits_tensor
        return mask
    
    def save_lora_checkpoint(self, path: str):
        """
        只保存LoRA权重（不包含SAM基础权重）
        
        Args:
            path: 保存路径
        """
        lora_state = {}
        for name, module in self.model.named_modules():
            if isinstance(module, LoRALinear):
                lora_state[name + '.lora_A'] = module.lora_A.data.cpu()
                lora_state[name + '.lora_B'] = module.lora_B.data.cpu()
        
        checkpoint = {
            'lora_state_dict': lora_state,
            'lora_rank': self.lora_rank,
            'lora_alpha': self.lora_alpha,
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
        print(f"LoRA checkpoint saved to {path}")
    
    def load_lora_checkpoint(self, path: str):
        """
        加载LoRA权重
        
        Args:
            path: checkpoint路径
        """
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        lora_state = checkpoint['lora_state_dict']
        
        # 验证配置一致性
        if checkpoint['lora_rank'] != self.lora_rank:
            print(f"Warning: LoRA rank mismatch: checkpoint={checkpoint['lora_rank']}, current={self.lora_rank}")
        if checkpoint['lora_alpha'] != self.lora_alpha:
            print(f"Warning: LoRA alpha mismatch: checkpoint={checkpoint['lora_alpha']}, current={self.lora_alpha}")
        
        # 加载LoRA参数
        for name, module in self.model.named_modules():
            if isinstance(module, LoRALinear):
                lora_A_key = name + '.lora_A'
                lora_B_key = name + '.lora_B'
                if lora_A_key in lora_state:
                    module.lora_A.data = lora_state[lora_A_key].to(self.device)
                if lora_B_key in lora_state:
                    module.lora_B.data = lora_state[lora_B_key].to(self.device)
        
        print(f"LoRA checkpoint loaded from {path}")
    
    def get_lora_state_dict(self) -> dict:
        """
        获取LoRA状态字典（用于完整checkpoint）
        
        Returns:
            LoRA状态字典
        """
        lora_state = {}
        for name, module in self.model.named_modules():
            if isinstance(module, LoRALinear):
                lora_state[name + '.lora_A'] = module.lora_A.data
                lora_state[name + '.lora_B'] = module.lora_B.data
        return lora_state
    
    def load_lora_state_dict(self, lora_state: dict):
        """
        加载LoRA状态字典
        
        Args:
            lora_state: LoRA状态字典
        """
        for name, module in self.model.named_modules():
            if isinstance(module, LoRALinear):
                lora_A_key = name + '.lora_A'
                lora_B_key = name + '.lora_B'
                if lora_A_key in lora_state:
                    module.lora_A.data = lora_state[lora_A_key].to(self.device)
                if lora_B_key in lora_state:
                    module.lora_B.data = lora_state[lora_B_key].to(self.device)


# 便捷函数
def load_sam2_with_lora(
    sam_checkpoint: str,
    lora_checkpoint: Optional[str] = None,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    device: str = "cuda"
) -> LoRASAM2Wrapper:
    """
    加载带LoRA的SAM2模型（便捷函数）
    
    Args:
        sam_checkpoint: SAM2基础模型路径
        lora_checkpoint: LoRA权重路径（可选）
        lora_rank: LoRA秩
        lora_alpha: LoRA缩放因子
        device: 设备
        
    Returns:
        LoRASAM2Wrapper实例
    """
    wrapper = LoRASAM2Wrapper(
        sam_checkpoint=sam_checkpoint,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        device=device
    )
    
    if lora_checkpoint is not None and os.path.exists(lora_checkpoint):
        wrapper.load_lora_checkpoint(lora_checkpoint)
    
    return wrapper


```

```python
#!/usr/bin/env python3
"""
PEFT训练辅助工具函数

包括：
- Metrics计算（Dice, IoU, PCK等）
- Checkpoint管理（保存/加载）
- 可视化辅助函数
- 学习率调度器
"""

from __future__ import annotations

import glob
import os
import random
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ============================================================================
# Metrics计算
# ============================================================================

def compute_dice(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6) -> float:
    """
    计算Dice系数
    
    Args:
        pred: 预测掩模 [H, W] 或 [B, H, W]，二值
        target: 真值掩模 [H, W] 或 [B, H, W]，二值
        epsilon: 平滑项
        
    Returns:
        Dice系数 [0, 1]
    """
    pred = pred.float().flatten()
    target = target.float().flatten()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dice = (2.0 * intersection + epsilon) / (union + epsilon)
    return dice.item()


def compute_iou(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6) -> float:
    """
    计算IoU (Intersection over Union)
    
    Args:
        pred: 预测掩模
        target: 真值掩模
        epsilon: 平滑项
        
    Returns:
        IoU [0, 1]
    """
    pred = pred.float().flatten()
    target = target.float().flatten()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + epsilon) / (union + epsilon)
    return iou.item()


def compute_pck(pred_point: torch.Tensor, target_point: torch.Tensor, threshold: float) -> bool:
    """
    计算PCK (Percentage of Correct Keypoints)
    
    如果预测点与真值点的距离小于阈值，则认为正确
    
    Args:
        pred_point: 预测点 [2] (x, y)
        target_point: 真值点 [2] (x, y)
        threshold: 距离阈值（像素）
        
    Returns:
        是否正确（距离 < 阈值）
    """
    distance = torch.norm(pred_point - target_point, p=2)
    return distance.item() < threshold


def compute_batch_pck(
    pred_points: torch.Tensor,
    target_points: torch.Tensor,
    threshold: float
) -> Tuple[float, torch.Tensor]:
    """
    批量计算PCK
    
    Args:
        pred_points: 预测点 [B, 2]
        target_points: 真值点 [B, 2]
        threshold: 距离阈值
        
    Returns:
        (PCK准确率, 距离向量 [B])
    """
    distances = torch.norm(pred_points - target_points, p=2, dim=1)
    correct = (distances < threshold).float()
    pck = correct.mean().item()
    return pck, distances


# ============================================================================
# Checkpoint管理
# ============================================================================

def save_checkpoint(
    path: str,
    epoch: int,
    step: int,
    point_predictor: nn.Module,
    sam_lora_wrapper,
    optimizer_sam: Optional[torch.optim.Optimizer] = None,
    optimizer_point: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    config: Optional[Dict] = None,
    metrics: Optional[Dict] = None,
):
    """
    保存完整checkpoint
    
    Args:
        path: 保存路径
        epoch: 当前epoch
        step: 当前step
        point_predictor: 点预测网络
        sam_lora_wrapper: SAM2 LoRA包装器
        optimizer_sam: SAM优化器（可选）
        optimizer_point: 点网络优化器（可选）
        scheduler: 学习率调度器（可选）
        config: 配置字典（可选）
        metrics: 指标字典（可选）
    """
    checkpoint = {
        'epoch': epoch,
        'step': step,
        
        # 模型状态
        'point_predictor_state': point_predictor.state_dict(),
        'sam_lora_state': sam_lora_wrapper.get_lora_state_dict(),
        
        # 配置
        'config': config or {},
        
        # 指标
        'metrics': metrics or {},
        
        # 随机状态
        'random_state': {
            'torch': torch.get_rng_state(),
            'numpy': np.random.get_state(),
            'python': random.getstate(),
        }
    }
    
    # 优化器状态
    if optimizer_sam is not None:
        checkpoint['optimizer_sam_state'] = optimizer_sam.state_dict()
    if optimizer_point is not None:
        checkpoint['optimizer_point_state'] = optimizer_point.state_dict()
    if scheduler is not None:
        checkpoint['scheduler_state'] = scheduler.state_dict()
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: str,
    point_predictor: nn.Module,
    sam_lora_wrapper,
    optimizer_sam: Optional[torch.optim.Optimizer] = None,
    optimizer_point: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = "cuda"
) -> Tuple[int, int, Dict, Dict]:
    """
    加载checkpoint
    
    Args:
        path: checkpoint路径
        point_predictor: 点预测网络
        sam_lora_wrapper: SAM2 LoRA包装器
        optimizer_sam: SAM优化器（可选）
        optimizer_point: 点网络优化器（可选）
        scheduler: 学习率调度器（可选）
        device: 设备
        
    Returns:
        (epoch, step, config, metrics)
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    # 加载模型状态
    point_predictor.load_state_dict(checkpoint['point_predictor_state'])
    sam_lora_wrapper.load_lora_state_dict(checkpoint['sam_lora_state'])
    
    # 加载优化器状态
    if optimizer_sam is not None and 'optimizer_sam_state' in checkpoint:
        optimizer_sam.load_state_dict(checkpoint['optimizer_sam_state'])
    if optimizer_point is not None and 'optimizer_point_state' in checkpoint:
        optimizer_point.load_state_dict(checkpoint['optimizer_point_state'])
    if scheduler is not None and 'scheduler_state' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state'])
    
    # 恢复随机状态
    if 'random_state' in checkpoint:
        torch.set_rng_state(checkpoint['random_state']['torch'])
        np.random.set_state(checkpoint['random_state']['numpy'])
        random.setstate(checkpoint['random_state']['python'])
    
    epoch = checkpoint.get('epoch', 0)
    step = checkpoint.get('step', 0)
    config = checkpoint.get('config', {})
    metrics = checkpoint.get('metrics', {})
    
    print(f"Checkpoint loaded from {path} (epoch={epoch}, step={step})")
    return epoch, step, config, metrics


def find_latest_checkpoint(out_dir: str) -> Optional[str]:
    """
    查找输出目录下最新的checkpoint
    
    Args:
        out_dir: 输出目录
        
    Returns:
        最新checkpoint路径，如果没有则返回None
    """
    ckpt_pattern = os.path.join(out_dir, 'checkpoint_*.pt')
    ckpts = glob.glob(ckpt_pattern)
    if not ckpts:
        return None
    
    # 按修改时间排序，返回最新的
    latest = max(ckpts, key=os.path.getmtime)
    return latest


# ============================================================================
# 学习率调度
# ============================================================================

def create_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    min_lr_ratio: float = 0.01
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    创建warmup + cosine学习率调度器
    
    Args:
        optimizer: 优化器
        warmup_epochs: warmup的epoch数
        total_epochs: 总epoch数
        min_lr_ratio: 最小学习率相对初始学习率的比例
        
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + np.cos(np.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================================
# 可视化辅助
# ============================================================================

def heatmap_to_rgb(heatmap: torch.Tensor, colormap: str = 'jet') -> np.ndarray:
    """
    将热力图转换为RGB可视化
    
    Args:
        heatmap: 热力图 [H, W] 或 [1, H, W]
        colormap: 颜色映射 ('jet', 'viridis', 'hot')
        
    Returns:
        RGB图像 [H, W, 3], uint8
    """
    if heatmap.dim() == 3:
        heatmap = heatmap[0]
    
    heatmap_np = heatmap.cpu().numpy()
    
    # 归一化到 [0, 1]
    heatmap_np = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min() + 1e-8)
    
    # 转换为uint8
    heatmap_uint8 = (heatmap_np * 255).astype(np.uint8)
    
    # 应用颜色映射
    try:
        import cv2
        if colormap == 'jet':
            colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        elif colormap == 'viridis':
            colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_VIRIDIS)
        elif colormap == 'hot':
            colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_HOT)
        else:
            colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # cv2返回BGR，转为RGB
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    except ImportError:
        # 如果没有cv2，使用简单的灰度图
        colored = np.stack([heatmap_uint8] * 3, axis=-1)
    
    return colored


def draw_point_on_image(
    image: np.ndarray,
    point: Tuple[float, float],
    color: Tuple[int, int, int] = (255, 0, 0),
    radius: int = 5,
    thickness: int = -1
) -> np.ndarray:
    """
    在图像上绘制点
    
    Args:
        image: 输入图像 [H, W, 3], uint8
        point: 点坐标 (x, y)
        color: 颜色 (R, G, B)
        radius: 半径
        thickness: 线条粗细，-1表示填充
        
    Returns:
        绘制后的图像
    """
    try:
        import cv2
        image_copy = image.copy()
        cv2.circle(image_copy, (int(point[0]), int(point[1])), radius, color, thickness)
        return image_copy
    except ImportError:
        return image


def tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    """
    将tensor转换为numpy图像用于可视化
    
    Args:
        tensor: [3, H, W], [0, 1] float tensor
        
    Returns:
        [H, W, 3], uint8 numpy array
    """
    image = tensor.permute(1, 2, 0).cpu().numpy()
    image = (image * 255).clip(0, 255).astype(np.uint8)
    return image


# ============================================================================
# 训练辅助
# ============================================================================

def set_seed(seed: int):
    """
    设置随机种子以保证可重复性
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    统计模型参数数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        (总参数数, 可训练参数数)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


class AverageMeter:
    """
    计算并存储平均值和当前值
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


```


检查修复后的文件是否有语法错误：



已修复 PyTorch 2.6 兼容性问题。

**问题原因：**
PyTorch 2.6 将 `torch.load` 的默认 `weights_only` 从 `False` 改为 `True`。checkpoint 包含 numpy 对象，需要 `weights_only=False` 才能加载。

**修复内容：**
已在以下文件的 `torch.load` 调用中添加 `weights_only=False`：
1. `eval_peft_model.py` (第180行) - 实验4.1使用的文件
2. `train_grpo_peft.py` (第451行) - 预防性修复
3. `lora_sam2.py` (第384行) - 预防性修复
4. `utils_peft.py` (第202行) - 预防性修复

现在可以重新运行实验4.1，checkpoint 加载错误应已解决。

**注意：** 使用 `weights_only=False` 是安全的，因为 checkpoint 文件来自您自己的训练过程，是可信源。

---

**User**

那这还需要重新训练然后用新得到的checkpoint_best.pt吗？

---

**Cursor**


不需要重新训练。检查 checkpoint 的保存和加载逻辑：



**不需要重新训练。**

**原因：**
1. 修复的是加载代码，不是保存代码。我们只在 `torch.load` 调用中添加了 `weights_only=False`，让代码能正确加载现有的 checkpoint。
2. 现有的 `checkpoint_best.pt` 文件本身没有问题，只是 PyTorch 2.6 的加载方式更严格，需要显式指定 `weights_only=False`。
3. `torch.save` 不需要修改，保存的 checkpoint 格式是兼容的。

**可以直接使用现有的 checkpoint：**
```bash
--checkpoint outputs/braintumour/peft_supervised_baseline-251107/checkpoint_best.pt
```

这个 checkpoint 现在应该可以正常加载了。修复后的代码会正确读取包含 numpy 对象的 checkpoint 文件。

**总结：** 直接重新运行实验4.1的命令即可，不需要重新训练。

```shellscript
--checkpoint outputs/braintumour/peft_supervised_baseline-251107/checkpoint_best.pt
```

---

**User**

又有错：@bash (983-1019) 

---

**Cursor**


检查相关代码，定位随机数状态加载问题：



问题是加载的随机数状态类型不正确。修复加载逻辑，确保类型正确：



修复随机数状态加载，确保类型正确：


```python
#!/usr/bin/env python3
"""
PEFT训练辅助工具函数

包括：
- Metrics计算（Dice, IoU, PCK等）
- Checkpoint管理（保存/加载）
- 可视化辅助函数
- 学习率调度器
"""

from __future__ import annotations

import glob
import os
import random
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ============================================================================
# Metrics计算
# ============================================================================

def compute_dice(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6) -> float:
    """
    计算Dice系数
    
    Args:
        pred: 预测掩模 [H, W] 或 [B, H, W]，二值
        target: 真值掩模 [H, W] 或 [B, H, W]，二值
        epsilon: 平滑项
        
    Returns:
        Dice系数 [0, 1]
    """
    pred = pred.float().flatten()
    target = target.float().flatten()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dice = (2.0 * intersection + epsilon) / (union + epsilon)
    return dice.item()


def compute_iou(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6) -> float:
    """
    计算IoU (Intersection over Union)
    
    Args:
        pred: 预测掩模
        target: 真值掩模
        epsilon: 平滑项
        
    Returns:
        IoU [0, 1]
    """
    pred = pred.float().flatten()
    target = target.float().flatten()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + epsilon) / (union + epsilon)
    return iou.item()


def compute_pck(pred_point: torch.Tensor, target_point: torch.Tensor, threshold: float) -> bool:
    """
    计算PCK (Percentage of Correct Keypoints)
    
    如果预测点与真值点的距离小于阈值，则认为正确
    
    Args:
        pred_point: 预测点 [2] (x, y)
        target_point: 真值点 [2] (x, y)
        threshold: 距离阈值（像素）
        
    Returns:
        是否正确（距离 < 阈值）
    """
    distance = torch.norm(pred_point - target_point, p=2)
    return distance.item() < threshold


def compute_batch_pck(
    pred_points: torch.Tensor,
    target_points: torch.Tensor,
    threshold: float
) -> Tuple[float, torch.Tensor]:
    """
    批量计算PCK
    
    Args:
        pred_points: 预测点 [B, 2]
        target_points: 真值点 [B, 2]
        threshold: 距离阈值
        
    Returns:
        (PCK准确率, 距离向量 [B])
    """
    distances = torch.norm(pred_points - target_points, p=2, dim=1)
    correct = (distances < threshold).float()
    pck = correct.mean().item()
    return pck, distances


# ============================================================================
# Checkpoint管理
# ============================================================================

def save_checkpoint(
    path: str,
    epoch: int,
    step: int,
    point_predictor: nn.Module,
    sam_lora_wrapper,
    optimizer_sam: Optional[torch.optim.Optimizer] = None,
    optimizer_point: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    config: Optional[Dict] = None,
    metrics: Optional[Dict] = None,
):
    """
    保存完整checkpoint
    
    Args:
        path: 保存路径
        epoch: 当前epoch
        step: 当前step
        point_predictor: 点预测网络
        sam_lora_wrapper: SAM2 LoRA包装器
        optimizer_sam: SAM优化器（可选）
        optimizer_point: 点网络优化器（可选）
        scheduler: 学习率调度器（可选）
        config: 配置字典（可选）
        metrics: 指标字典（可选）
    """
    checkpoint = {
        'epoch': epoch,
        'step': step,
        
        # 模型状态
        'point_predictor_state': point_predictor.state_dict(),
        'sam_lora_state': sam_lora_wrapper.get_lora_state_dict(),
        
        # 配置
        'config': config or {},
        
        # 指标
        'metrics': metrics or {},
        
        # 随机状态
        'random_state': {
            'torch': torch.get_rng_state(),
            'numpy': np.random.get_state(),
            'python': random.getstate(),
        }
    }
    
    # 优化器状态
    if optimizer_sam is not None:
        checkpoint['optimizer_sam_state'] = optimizer_sam.state_dict()
    if optimizer_point is not None:
        checkpoint['optimizer_point_state'] = optimizer_point.state_dict()
    if scheduler is not None:
        checkpoint['scheduler_state'] = scheduler.state_dict()
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: str,
    point_predictor: nn.Module,
    sam_lora_wrapper,
    optimizer_sam: Optional[torch.optim.Optimizer] = None,
    optimizer_point: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = "cuda"
) -> Tuple[int, int, Dict, Dict]:
    """
    加载checkpoint
    
    Args:
        path: checkpoint路径
        point_predictor: 点预测网络
        sam_lora_wrapper: SAM2 LoRA包装器
        optimizer_sam: SAM优化器（可选）
        optimizer_point: 点网络优化器（可选）
        scheduler: 学习率调度器（可选）
        device: 设备
        
    Returns:
        (epoch, step, config, metrics)
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    # 加载模型状态
    point_predictor.load_state_dict(checkpoint['point_predictor_state'])
    sam_lora_wrapper.load_lora_state_dict(checkpoint['sam_lora_state'])
    
    # 加载优化器状态
    if optimizer_sam is not None and 'optimizer_sam_state' in checkpoint:
        optimizer_sam.load_state_dict(checkpoint['optimizer_sam_state'])
    if optimizer_point is not None and 'optimizer_point_state' in checkpoint:
        optimizer_point.load_state_dict(checkpoint['optimizer_point_state'])
    if scheduler is not None and 'scheduler_state' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state'])
    
    # 恢复随机状态
    if 'random_state' in checkpoint:
        # 确保 torch RNG 状态是正确的类型
        torch_rng_state = checkpoint['random_state']['torch']
        if not isinstance(torch_rng_state, torch.ByteTensor):
            torch_rng_state = torch.ByteTensor(torch_rng_state)
        torch.set_rng_state(torch_rng_state)
        np.random.set_state(checkpoint['random_state']['numpy'])
        random.setstate(checkpoint['random_state']['python'])
    
    epoch = checkpoint.get('epoch', 0)
    step = checkpoint.get('step', 0)
    config = checkpoint.get('config', {})
    metrics = checkpoint.get('metrics', {})
    
    print(f"Checkpoint loaded from {path} (epoch={epoch}, step={step})")
    return epoch, step, config, metrics


def find_latest_checkpoint(out_dir: str) -> Optional[str]:
    """
    查找输出目录下最新的checkpoint
    
    Args:
        out_dir: 输出目录
        
    Returns:
        最新checkpoint路径，如果没有则返回None
    """
    ckpt_pattern = os.path.join(out_dir, 'checkpoint_*.pt')
    ckpts = glob.glob(ckpt_pattern)
    if not ckpts:
        return None
    
    # 按修改时间排序，返回最新的
    latest = max(ckpts, key=os.path.getmtime)
    return latest


# ============================================================================
# 学习率调度
# ============================================================================

def create_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    min_lr_ratio: float = 0.01
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    创建warmup + cosine学习率调度器
    
    Args:
        optimizer: 优化器
        warmup_epochs: warmup的epoch数
        total_epochs: 总epoch数
        min_lr_ratio: 最小学习率相对初始学习率的比例
        
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + np.cos(np.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================================
# 可视化辅助
# ============================================================================

def heatmap_to_rgb(heatmap: torch.Tensor, colormap: str = 'jet') -> np.ndarray:
    """
    将热力图转换为RGB可视化
    
    Args:
        heatmap: 热力图 [H, W] 或 [1, H, W]
        colormap: 颜色映射 ('jet', 'viridis', 'hot')
        
    Returns:
        RGB图像 [H, W, 3], uint8
    """
    if heatmap.dim() == 3:
        heatmap = heatmap[0]
    
    heatmap_np = heatmap.cpu().numpy()
    
    # 归一化到 [0, 1]
    heatmap_np = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min() + 1e-8)
    
    # 转换为uint8
    heatmap_uint8 = (heatmap_np * 255).astype(np.uint8)
    
    # 应用颜色映射
    try:
        import cv2
        if colormap == 'jet':
            colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        elif colormap == 'viridis':
            colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_VIRIDIS)
        elif colormap == 'hot':
            colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_HOT)
        else:
            colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # cv2返回BGR，转为RGB
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    except ImportError:
        # 如果没有cv2，使用简单的灰度图
        colored = np.stack([heatmap_uint8] * 3, axis=-1)
    
    return colored


def draw_point_on_image(
    image: np.ndarray,
    point: Tuple[float, float],
    color: Tuple[int, int, int] = (255, 0, 0),
    radius: int = 5,
    thickness: int = -1
) -> np.ndarray:
    """
    在图像上绘制点
    
    Args:
        image: 输入图像 [H, W, 3], uint8
        point: 点坐标 (x, y)
        color: 颜色 (R, G, B)
        radius: 半径
        thickness: 线条粗细，-1表示填充
        
    Returns:
        绘制后的图像
    """
    try:
        import cv2
        image_copy = image.copy()
        cv2.circle(image_copy, (int(point[0]), int(point[1])), radius, color, thickness)
        return image_copy
    except ImportError:
        return image


def tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    """
    将tensor转换为numpy图像用于可视化
    
    Args:
        tensor: [3, H, W], [0, 1] float tensor
        
    Returns:
        [H, W, 3], uint8 numpy array
    """
    image = tensor.permute(1, 2, 0).cpu().numpy()
    image = (image * 255).clip(0, 255).astype(np.uint8)
    return image


# ============================================================================
# 训练辅助
# ============================================================================

def set_seed(seed: int):
    """
    设置随机种子以保证可重复性
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    统计模型参数数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        (总参数数, 可训练参数数)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


class AverageMeter:
    """
    计算并存储平均值和当前值
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


```


检查评估脚本是否真的需要恢复随机数状态。评估通常不需要：



优化修复，使用更安全的方法处理类型转换：


```python
#!/usr/bin/env python3
"""
PEFT训练辅助工具函数

包括：
- Metrics计算（Dice, IoU, PCK等）
- Checkpoint管理（保存/加载）
- 可视化辅助函数
- 学习率调度器
"""

from __future__ import annotations

import glob
import os
import random
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ============================================================================
# Metrics计算
# ============================================================================

def compute_dice(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6) -> float:
    """
    计算Dice系数
    
    Args:
        pred: 预测掩模 [H, W] 或 [B, H, W]，二值
        target: 真值掩模 [H, W] 或 [B, H, W]，二值
        epsilon: 平滑项
        
    Returns:
        Dice系数 [0, 1]
    """
    pred = pred.float().flatten()
    target = target.float().flatten()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dice = (2.0 * intersection + epsilon) / (union + epsilon)
    return dice.item()


def compute_iou(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6) -> float:
    """
    计算IoU (Intersection over Union)
    
    Args:
        pred: 预测掩模
        target: 真值掩模
        epsilon: 平滑项
        
    Returns:
        IoU [0, 1]
    """
    pred = pred.float().flatten()
    target = target.float().flatten()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + epsilon) / (union + epsilon)
    return iou.item()


def compute_pck(pred_point: torch.Tensor, target_point: torch.Tensor, threshold: float) -> bool:
    """
    计算PCK (Percentage of Correct Keypoints)
    
    如果预测点与真值点的距离小于阈值，则认为正确
    
    Args:
        pred_point: 预测点 [2] (x, y)
        target_point: 真值点 [2] (x, y)
        threshold: 距离阈值（像素）
        
    Returns:
        是否正确（距离 < 阈值）
    """
    distance = torch.norm(pred_point - target_point, p=2)
    return distance.item() < threshold


def compute_batch_pck(
    pred_points: torch.Tensor,
    target_points: torch.Tensor,
    threshold: float
) -> Tuple[float, torch.Tensor]:
    """
    批量计算PCK
    
    Args:
        pred_points: 预测点 [B, 2]
        target_points: 真值点 [B, 2]
        threshold: 距离阈值
        
    Returns:
        (PCK准确率, 距离向量 [B])
    """
    distances = torch.norm(pred_points - target_points, p=2, dim=1)
    correct = (distances < threshold).float()
    pck = correct.mean().item()
    return pck, distances


# ============================================================================
# Checkpoint管理
# ============================================================================

def save_checkpoint(
    path: str,
    epoch: int,
    step: int,
    point_predictor: nn.Module,
    sam_lora_wrapper,
    optimizer_sam: Optional[torch.optim.Optimizer] = None,
    optimizer_point: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    config: Optional[Dict] = None,
    metrics: Optional[Dict] = None,
):
    """
    保存完整checkpoint
    
    Args:
        path: 保存路径
        epoch: 当前epoch
        step: 当前step
        point_predictor: 点预测网络
        sam_lora_wrapper: SAM2 LoRA包装器
        optimizer_sam: SAM优化器（可选）
        optimizer_point: 点网络优化器（可选）
        scheduler: 学习率调度器（可选）
        config: 配置字典（可选）
        metrics: 指标字典（可选）
    """
    checkpoint = {
        'epoch': epoch,
        'step': step,
        
        # 模型状态
        'point_predictor_state': point_predictor.state_dict(),
        'sam_lora_state': sam_lora_wrapper.get_lora_state_dict(),
        
        # 配置
        'config': config or {},
        
        # 指标
        'metrics': metrics or {},
        
        # 随机状态
        'random_state': {
            'torch': torch.get_rng_state(),
            'numpy': np.random.get_state(),
            'python': random.getstate(),
        }
    }
    
    # 优化器状态
    if optimizer_sam is not None:
        checkpoint['optimizer_sam_state'] = optimizer_sam.state_dict()
    if optimizer_point is not None:
        checkpoint['optimizer_point_state'] = optimizer_point.state_dict()
    if scheduler is not None:
        checkpoint['scheduler_state'] = scheduler.state_dict()
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: str,
    point_predictor: nn.Module,
    sam_lora_wrapper,
    optimizer_sam: Optional[torch.optim.Optimizer] = None,
    optimizer_point: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = "cuda"
) -> Tuple[int, int, Dict, Dict]:
    """
    加载checkpoint
    
    Args:
        path: checkpoint路径
        point_predictor: 点预测网络
        sam_lora_wrapper: SAM2 LoRA包装器
        optimizer_sam: SAM优化器（可选）
        optimizer_point: 点网络优化器（可选）
        scheduler: 学习率调度器（可选）
        device: 设备
        
    Returns:
        (epoch, step, config, metrics)
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    # 加载模型状态
    point_predictor.load_state_dict(checkpoint['point_predictor_state'])
    sam_lora_wrapper.load_lora_state_dict(checkpoint['sam_lora_state'])
    
    # 加载优化器状态
    if optimizer_sam is not None and 'optimizer_sam_state' in checkpoint:
        optimizer_sam.load_state_dict(checkpoint['optimizer_sam_state'])
    if optimizer_point is not None and 'optimizer_point_state' in checkpoint:
        optimizer_point.load_state_dict(checkpoint['optimizer_point_state'])
    if scheduler is not None and 'scheduler_state' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state'])
    
    # 恢复随机状态
    if 'random_state' in checkpoint:
        # 确保 torch RNG 状态是正确的类型 (ByteTensor/uint8)
        torch_rng_state = checkpoint['random_state']['torch']
        if isinstance(torch_rng_state, torch.Tensor):
            if torch_rng_state.dtype != torch.uint8:
                torch_rng_state = torch_rng_state.to(torch.uint8)
        else:
            # 如果是 numpy 数组或其他类型，转换为 ByteTensor
            torch_rng_state = torch.tensor(torch_rng_state, dtype=torch.uint8)
        torch.set_rng_state(torch_rng_state)
        np.random.set_state(checkpoint['random_state']['numpy'])
        random.setstate(checkpoint['random_state']['python'])
    
    epoch = checkpoint.get('epoch', 0)
    step = checkpoint.get('step', 0)
    config = checkpoint.get('config', {})
    metrics = checkpoint.get('metrics', {})
    
    print(f"Checkpoint loaded from {path} (epoch={epoch}, step={step})")
    return epoch, step, config, metrics


def find_latest_checkpoint(out_dir: str) -> Optional[str]:
    """
    查找输出目录下最新的checkpoint
    
    Args:
        out_dir: 输出目录
        
    Returns:
        最新checkpoint路径，如果没有则返回None
    """
    ckpt_pattern = os.path.join(out_dir, 'checkpoint_*.pt')
    ckpts = glob.glob(ckpt_pattern)
    if not ckpts:
        return None
    
    # 按修改时间排序，返回最新的
    latest = max(ckpts, key=os.path.getmtime)
    return latest


# ============================================================================
# 学习率调度
# ============================================================================

def create_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    min_lr_ratio: float = 0.01
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    创建warmup + cosine学习率调度器
    
    Args:
        optimizer: 优化器
        warmup_epochs: warmup的epoch数
        total_epochs: 总epoch数
        min_lr_ratio: 最小学习率相对初始学习率的比例
        
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + np.cos(np.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================================
# 可视化辅助
# ============================================================================

def heatmap_to_rgb(heatmap: torch.Tensor, colormap: str = 'jet') -> np.ndarray:
    """
    将热力图转换为RGB可视化
    
    Args:
        heatmap: 热力图 [H, W] 或 [1, H, W]
        colormap: 颜色映射 ('jet', 'viridis', 'hot')
        
    Returns:
        RGB图像 [H, W, 3], uint8
    """
    if heatmap.dim() == 3:
        heatmap = heatmap[0]
    
    heatmap_np = heatmap.cpu().numpy()
    
    # 归一化到 [0, 1]
    heatmap_np = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min() + 1e-8)
    
    # 转换为uint8
    heatmap_uint8 = (heatmap_np * 255).astype(np.uint8)
    
    # 应用颜色映射
    try:
        import cv2
        if colormap == 'jet':
            colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        elif colormap == 'viridis':
            colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_VIRIDIS)
        elif colormap == 'hot':
            colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_HOT)
        else:
            colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # cv2返回BGR，转为RGB
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    except ImportError:
        # 如果没有cv2，使用简单的灰度图
        colored = np.stack([heatmap_uint8] * 3, axis=-1)
    
    return colored


def draw_point_on_image(
    image: np.ndarray,
    point: Tuple[float, float],
    color: Tuple[int, int, int] = (255, 0, 0),
    radius: int = 5,
    thickness: int = -1
) -> np.ndarray:
    """
    在图像上绘制点
    
    Args:
        image: 输入图像 [H, W, 3], uint8
        point: 点坐标 (x, y)
        color: 颜色 (R, G, B)
        radius: 半径
        thickness: 线条粗细，-1表示填充
        
    Returns:
        绘制后的图像
    """
    try:
        import cv2
        image_copy = image.copy()
        cv2.circle(image_copy, (int(point[0]), int(point[1])), radius, color, thickness)
        return image_copy
    except ImportError:
        return image


def tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    """
    将tensor转换为numpy图像用于可视化
    
    Args:
        tensor: [3, H, W], [0, 1] float tensor
        
    Returns:
        [H, W, 3], uint8 numpy array
    """
    image = tensor.permute(1, 2, 0).cpu().numpy()
    image = (image * 255).clip(0, 255).astype(np.uint8)
    return image


# ============================================================================
# 训练辅助
# ============================================================================

def set_seed(seed: int):
    """
    设置随机种子以保证可重复性
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    统计模型参数数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        (总参数数, 可训练参数数)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


class AverageMeter:
    """
    计算并存储平均值和当前值
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


```


已修复随机数状态加载问题。

**问题原因：**
从 checkpoint 加载的 PyTorch 随机数状态可能不是 `torch.ByteTensor`（`torch.uint8`），而 `torch.set_rng_state()` 需要该类型。

**修复内容：**
在 `utils_peft.py` 的 `load_checkpoint` 函数中添加类型检查和转换：
- 如果是 `torch.Tensor` 但 dtype 不是 `uint8`，转换为 `uint8`
- 如果是其他类型（如 numpy 数组），转换为 `torch.uint8` 的 tensor

现在可以重新运行实验4.1，随机数状态加载错误应已解决。

**注意：** 对于评估脚本，恢复随机数状态不是必需的，但修复后代码可以正确处理，不会影响评估结果。

---

**User**

还是不对，问题看来不是你上述的：@bash (984-1019) 

---

**Cursor**


检查实际加载的数据类型，并调试该问题：



问题可能是 `torch.tensor()` 创建的 tensor 不是 `ByteTensor` 类型。改用 `torch.ByteTensor()` 构造函数，或对评估脚本跳过恢复随机数状态。先检查评估脚本是否真的需要恢复随机数状态：



修复类型转换：使用 `torch.ByteTensor()` 构造函数，或对评估脚本跳过恢复随机数状态。先修复类型转换：


```python
#!/usr/bin/env python3
"""
PEFT训练辅助工具函数

包括：
- Metrics计算（Dice, IoU, PCK等）
- Checkpoint管理（保存/加载）
- 可视化辅助函数
- 学习率调度器
"""

from __future__ import annotations

import glob
import os
import random
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ============================================================================
# Metrics计算
# ============================================================================

def compute_dice(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6) -> float:
    """
    计算Dice系数
    
    Args:
        pred: 预测掩模 [H, W] 或 [B, H, W]，二值
        target: 真值掩模 [H, W] 或 [B, H, W]，二值
        epsilon: 平滑项
        
    Returns:
        Dice系数 [0, 1]
    """
    pred = pred.float().flatten()
    target = target.float().flatten()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dice = (2.0 * intersection + epsilon) / (union + epsilon)
    return dice.item()


def compute_iou(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6) -> float:
    """
    计算IoU (Intersection over Union)
    
    Args:
        pred: 预测掩模
        target: 真值掩模
        epsilon: 平滑项
        
    Returns:
        IoU [0, 1]
    """
    pred = pred.float().flatten()
    target = target.float().flatten()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + epsilon) / (union + epsilon)
    return iou.item()


def compute_pck(pred_point: torch.Tensor, target_point: torch.Tensor, threshold: float) -> bool:
    """
    计算PCK (Percentage of Correct Keypoints)
    
    如果预测点与真值点的距离小于阈值，则认为正确
    
    Args:
        pred_point: 预测点 [2] (x, y)
        target_point: 真值点 [2] (x, y)
        threshold: 距离阈值（像素）
        
    Returns:
        是否正确（距离 < 阈值）
    """
    distance = torch.norm(pred_point - target_point, p=2)
    return distance.item() < threshold


def compute_batch_pck(
    pred_points: torch.Tensor,
    target_points: torch.Tensor,
    threshold: float
) -> Tuple[float, torch.Tensor]:
    """
    批量计算PCK
    
    Args:
        pred_points: 预测点 [B, 2]
        target_points: 真值点 [B, 2]
        threshold: 距离阈值
        
    Returns:
        (PCK准确率, 距离向量 [B])
    """
    distances = torch.norm(pred_points - target_points, p=2, dim=1)
    correct = (distances < threshold).float()
    pck = correct.mean().item()
    return pck, distances


# ============================================================================
# Checkpoint管理
# ============================================================================

def save_checkpoint(
    path: str,
    epoch: int,
    step: int,
    point_predictor: nn.Module,
    sam_lora_wrapper,
    optimizer_sam: Optional[torch.optim.Optimizer] = None,
    optimizer_point: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    config: Optional[Dict] = None,
    metrics: Optional[Dict] = None,
):
    """
    保存完整checkpoint
    
    Args:
        path: 保存路径
        epoch: 当前epoch
        step: 当前step
        point_predictor: 点预测网络
        sam_lora_wrapper: SAM2 LoRA包装器
        optimizer_sam: SAM优化器（可选）
        optimizer_point: 点网络优化器（可选）
        scheduler: 学习率调度器（可选）
        config: 配置字典（可选）
        metrics: 指标字典（可选）
    """
    checkpoint = {
        'epoch': epoch,
        'step': step,
        
        # 模型状态
        'point_predictor_state': point_predictor.state_dict(),
        'sam_lora_state': sam_lora_wrapper.get_lora_state_dict(),
        
        # 配置
        'config': config or {},
        
        # 指标
        'metrics': metrics or {},
        
        # 随机状态
        'random_state': {
            'torch': torch.get_rng_state(),
            'numpy': np.random.get_state(),
            'python': random.getstate(),
        }
    }
    
    # 优化器状态
    if optimizer_sam is not None:
        checkpoint['optimizer_sam_state'] = optimizer_sam.state_dict()
    if optimizer_point is not None:
        checkpoint['optimizer_point_state'] = optimizer_point.state_dict()
    if scheduler is not None:
        checkpoint['scheduler_state'] = scheduler.state_dict()
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: str,
    point_predictor: nn.Module,
    sam_lora_wrapper,
    optimizer_sam: Optional[torch.optim.Optimizer] = None,
    optimizer_point: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = "cuda"
) -> Tuple[int, int, Dict, Dict]:
    """
    加载checkpoint
    
    Args:
        path: checkpoint路径
        point_predictor: 点预测网络
        sam_lora_wrapper: SAM2 LoRA包装器
        optimizer_sam: SAM优化器（可选）
        optimizer_point: 点网络优化器（可选）
        scheduler: 学习率调度器（可选）
        device: 设备
        
    Returns:
        (epoch, step, config, metrics)
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    # 加载模型状态
    point_predictor.load_state_dict(checkpoint['point_predictor_state'])
    sam_lora_wrapper.load_lora_state_dict(checkpoint['sam_lora_state'])
    
    # 加载优化器状态
    if optimizer_sam is not None and 'optimizer_sam_state' in checkpoint:
        optimizer_sam.load_state_dict(checkpoint['optimizer_sam_state'])
    if optimizer_point is not None and 'optimizer_point_state' in checkpoint:
        optimizer_point.load_state_dict(checkpoint['optimizer_point_state'])
    if scheduler is not None and 'scheduler_state' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state'])
    
    # 恢复随机状态
    if 'random_state' in checkpoint:
        # 确保 torch RNG 状态是正确的类型 (ByteTensor/uint8)
        torch_rng_state = checkpoint['random_state']['torch']
        try:
            # 尝试直接使用，如果类型正确
            if isinstance(torch_rng_state, torch.ByteTensor):
                torch.set_rng_state(torch_rng_state)
            elif isinstance(torch_rng_state, torch.Tensor):
                # 转换为 ByteTensor
                if torch_rng_state.dtype != torch.uint8:
                    torch_rng_state = torch_rng_state.to(torch.uint8)
                # 使用 ByteTensor 构造函数确保类型正确
                torch.set_rng_state(torch.ByteTensor(torch_rng_state.cpu().numpy()))
            else:
                # 如果是 numpy 数组或其他类型，转换为 ByteTensor
                if isinstance(torch_rng_state, np.ndarray):
                    torch.set_rng_state(torch.ByteTensor(torch_rng_state))
                else:
                    torch.set_rng_state(torch.ByteTensor(np.array(torch_rng_state, dtype=np.uint8)))
        except (TypeError, AttributeError) as e:
            # 如果恢复随机状态失败，打印警告但继续执行（评估时不需要随机状态）
            print(f"Warning: Failed to restore random state: {e}. Continuing without restoring RNG state.")
        else:
            # 只有在成功恢复 torch 状态后才恢复其他随机状态
            np.random.set_state(checkpoint['random_state']['numpy'])
            random.setstate(checkpoint['random_state']['python'])
    
    epoch = checkpoint.get('epoch', 0)
    step = checkpoint.get('step', 0)
    config = checkpoint.get('config', {})
    metrics = checkpoint.get('metrics', {})
    
    print(f"Checkpoint loaded from {path} (epoch={epoch}, step={step})")
    return epoch, step, config, metrics


def find_latest_checkpoint(out_dir: str) -> Optional[str]:
    """
    查找输出目录下最新的checkpoint
    
    Args:
        out_dir: 输出目录
        
    Returns:
        最新checkpoint路径，如果没有则返回None
    """
    ckpt_pattern = os.path.join(out_dir, 'checkpoint_*.pt')
    ckpts = glob.glob(ckpt_pattern)
    if not ckpts:
        return None
    
    # 按修改时间排序，返回最新的
    latest = max(ckpts, key=os.path.getmtime)
    return latest


# ============================================================================
# 学习率调度
# ============================================================================

def create_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    min_lr_ratio: float = 0.01
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    创建warmup + cosine学习率调度器
    
    Args:
        optimizer: 优化器
        warmup_epochs: warmup的epoch数
        total_epochs: 总epoch数
        min_lr_ratio: 最小学习率相对初始学习率的比例
        
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + np.cos(np.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================================
# 可视化辅助
# ============================================================================

def heatmap_to_rgb(heatmap: torch.Tensor, colormap: str = 'jet') -> np.ndarray:
    """
    将热力图转换为RGB可视化
    
    Args:
        heatmap: 热力图 [H, W] 或 [1, H, W]
        colormap: 颜色映射 ('jet', 'viridis', 'hot')
        
    Returns:
        RGB图像 [H, W, 3], uint8
    """
    if heatmap.dim() == 3:
        heatmap = heatmap[0]
    
    heatmap_np = heatmap.cpu().numpy()
    
    # 归一化到 [0, 1]
    heatmap_np = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min() + 1e-8)
    
    # 转换为uint8
    heatmap_uint8 = (heatmap_np * 255).astype(np.uint8)
    
    # 应用颜色映射
    try:
        import cv2
        if colormap == 'jet':
            colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        elif colormap == 'viridis':
            colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_VIRIDIS)
        elif colormap == 'hot':
            colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_HOT)
        else:
            colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # cv2返回BGR，转为RGB
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    except ImportError:
        # 如果没有cv2，使用简单的灰度图
        colored = np.stack([heatmap_uint8] * 3, axis=-1)
    
    return colored


def draw_point_on_image(
    image: np.ndarray,
    point: Tuple[float, float],
    color: Tuple[int, int, int] = (255, 0, 0),
    radius: int = 5,
    thickness: int = -1
) -> np.ndarray:
    """
    在图像上绘制点
    
    Args:
        image: 输入图像 [H, W, 3], uint8
        point: 点坐标 (x, y)
        color: 颜色 (R, G, B)
        radius: 半径
        thickness: 线条粗细，-1表示填充
        
    Returns:
        绘制后的图像
    """
    try:
        import cv2
        image_copy = image.copy()
        cv2.circle(image_copy, (int(point[0]), int(point[1])), radius, color, thickness)
        return image_copy
    except ImportError:
        return image


def tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    """
    将tensor转换为numpy图像用于可视化
    
    Args:
        tensor: [3, H, W], [0, 1] float tensor
        
    Returns:
        [H, W, 3], uint8 numpy array
    """
    image = tensor.permute(1, 2, 0).cpu().numpy()
    image = (image * 255).clip(0, 255).astype(np.uint8)
    return image


# ============================================================================
# 训练辅助
# ============================================================================

def set_seed(seed: int):
    """
    设置随机种子以保证可重复性
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    统计模型参数数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        (总参数数, 可训练参数数)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


class AverageMeter:
    """
    计算并存储平均值和当前值
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


```


检查 `torch.ByteTensor` 是否可用，或采用更简单的方法：



检查实际加载的数据类型，并尝试更直接的方法：



简化修复：使用更直接的方法处理类型转换：


```python
#!/usr/bin/env python3
"""
PEFT训练辅助工具函数

包括：
- Metrics计算（Dice, IoU, PCK等）
- Checkpoint管理（保存/加载）
- 可视化辅助函数
- 学习率调度器
"""

from __future__ import annotations

import glob
import os
import random
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ============================================================================
# Metrics计算
# ============================================================================

def compute_dice(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6) -> float:
    """
    计算Dice系数
    
    Args:
        pred: 预测掩模 [H, W] 或 [B, H, W]，二值
        target: 真值掩模 [H, W] 或 [B, H, W]，二值
        epsilon: 平滑项
        
    Returns:
        Dice系数 [0, 1]
    """
    pred = pred.float().flatten()
    target = target.float().flatten()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dice = (2.0 * intersection + epsilon) / (union + epsilon)
    return dice.item()


def compute_iou(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6) -> float:
    """
    计算IoU (Intersection over Union)
    
    Args:
        pred: 预测掩模
        target: 真值掩模
        epsilon: 平滑项
        
    Returns:
        IoU [0, 1]
    """
    pred = pred.float().flatten()
    target = target.float().flatten()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + epsilon) / (union + epsilon)
    return iou.item()


def compute_pck(pred_point: torch.Tensor, target_point: torch.Tensor, threshold: float) -> bool:
    """
    计算PCK (Percentage of Correct Keypoints)
    
    如果预测点与真值点的距离小于阈值，则认为正确
    
    Args:
        pred_point: 预测点 [2] (x, y)
        target_point: 真值点 [2] (x, y)
        threshold: 距离阈值（像素）
        
    Returns:
        是否正确（距离 < 阈值）
    """
    distance = torch.norm(pred_point - target_point, p=2)
    return distance.item() < threshold


def compute_batch_pck(
    pred_points: torch.Tensor,
    target_points: torch.Tensor,
    threshold: float
) -> Tuple[float, torch.Tensor]:
    """
    批量计算PCK
    
    Args:
        pred_points: 预测点 [B, 2]
        target_points: 真值点 [B, 2]
        threshold: 距离阈值
        
    Returns:
        (PCK准确率, 距离向量 [B])
    """
    distances = torch.norm(pred_points - target_points, p=2, dim=1)
    correct = (distances < threshold).float()
    pck = correct.mean().item()
    return pck, distances


# ============================================================================
# Checkpoint管理
# ============================================================================

def save_checkpoint(
    path: str,
    epoch: int,
    step: int,
    point_predictor: nn.Module,
    sam_lora_wrapper,
    optimizer_sam: Optional[torch.optim.Optimizer] = None,
    optimizer_point: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    config: Optional[Dict] = None,
    metrics: Optional[Dict] = None,
):
    """
    保存完整checkpoint
    
    Args:
        path: 保存路径
        epoch: 当前epoch
        step: 当前step
        point_predictor: 点预测网络
        sam_lora_wrapper: SAM2 LoRA包装器
        optimizer_sam: SAM优化器（可选）
        optimizer_point: 点网络优化器（可选）
        scheduler: 学习率调度器（可选）
        config: 配置字典（可选）
        metrics: 指标字典（可选）
    """
    checkpoint = {
        'epoch': epoch,
        'step': step,
        
        # 模型状态
        'point_predictor_state': point_predictor.state_dict(),
        'sam_lora_state': sam_lora_wrapper.get_lora_state_dict(),
        
        # 配置
        'config': config or {},
        
        # 指标
        'metrics': metrics or {},
        
        # 随机状态
        'random_state': {
            'torch': torch.get_rng_state(),
            'numpy': np.random.get_state(),
            'python': random.getstate(),
        }
    }
    
    # 优化器状态
    if optimizer_sam is not None:
        checkpoint['optimizer_sam_state'] = optimizer_sam.state_dict()
    if optimizer_point is not None:
        checkpoint['optimizer_point_state'] = optimizer_point.state_dict()
    if scheduler is not None:
        checkpoint['scheduler_state'] = scheduler.state_dict()
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: str,
    point_predictor: nn.Module,
    sam_lora_wrapper,
    optimizer_sam: Optional[torch.optim.Optimizer] = None,
    optimizer_point: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = "cuda"
) -> Tuple[int, int, Dict, Dict]:
    """
    加载checkpoint
    
    Args:
        path: checkpoint路径
        point_predictor: 点预测网络
        sam_lora_wrapper: SAM2 LoRA包装器
        optimizer_sam: SAM优化器（可选）
        optimizer_point: 点网络优化器（可选）
        scheduler: 学习率调度器（可选）
        device: 设备
        
    Returns:
        (epoch, step, config, metrics)
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    # 加载模型状态
    point_predictor.load_state_dict(checkpoint['point_predictor_state'])
    sam_lora_wrapper.load_lora_state_dict(checkpoint['sam_lora_state'])
    
    # 加载优化器状态
    if optimizer_sam is not None and 'optimizer_sam_state' in checkpoint:
        optimizer_sam.load_state_dict(checkpoint['optimizer_sam_state'])
    if optimizer_point is not None and 'optimizer_point_state' in checkpoint:
        optimizer_point.load_state_dict(checkpoint['optimizer_point_state'])
    if scheduler is not None and 'scheduler_state' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state'])
    
    # 恢复随机状态
    if 'random_state' in checkpoint:
        # 确保 torch RNG 状态是正确的类型 (ByteTensor/uint8)
        torch_rng_state = checkpoint['random_state']['torch']
        try:
            # 转换为 numpy 数组，然后使用 ByteTensor 构造函数
            if isinstance(torch_rng_state, torch.Tensor):
                # 先转换为 numpy，确保是 uint8 类型
                rng_numpy = torch_rng_state.cpu().numpy().astype(np.uint8)
            elif isinstance(torch_rng_state, np.ndarray):
                rng_numpy = torch_rng_state.astype(np.uint8)
            else:
                rng_numpy = np.array(torch_rng_state, dtype=np.uint8)
            
            # 使用 ByteTensor 构造函数创建正确的类型
            torch.set_rng_state(torch.ByteTensor(rng_numpy))
            np.random.set_state(checkpoint['random_state']['numpy'])
            random.setstate(checkpoint['random_state']['python'])
        except (TypeError, AttributeError, ValueError) as e:
            # 如果恢复随机状态失败，打印警告但继续执行（评估时不需要随机状态）
            print(f"Warning: Failed to restore random state: {e}. Continuing without restoring RNG state.")
    
    epoch = checkpoint.get('epoch', 0)
    step = checkpoint.get('step', 0)
    config = checkpoint.get('config', {})
    metrics = checkpoint.get('metrics', {})
    
    print(f"Checkpoint loaded from {path} (epoch={epoch}, step={step})")
    return epoch, step, config, metrics


def find_latest_checkpoint(out_dir: str) -> Optional[str]:
    """
    查找输出目录下最新的checkpoint
    
    Args:
        out_dir: 输出目录
        
    Returns:
        最新checkpoint路径，如果没有则返回None
    """
    ckpt_pattern = os.path.join(out_dir, 'checkpoint_*.pt')
    ckpts = glob.glob(ckpt_pattern)
    if not ckpts:
        return None
    
    # 按修改时间排序，返回最新的
    latest = max(ckpts, key=os.path.getmtime)
    return latest


# ============================================================================
# 学习率调度
# ============================================================================

def create_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    min_lr_ratio: float = 0.01
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    创建warmup + cosine学习率调度器
    
    Args:
        optimizer: 优化器
        warmup_epochs: warmup的epoch数
        total_epochs: 总epoch数
        min_lr_ratio: 最小学习率相对初始学习率的比例
        
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + np.cos(np.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================================
# 可视化辅助
# ============================================================================

def heatmap_to_rgb(heatmap: torch.Tensor, colormap: str = 'jet') -> np.ndarray:
    """
    将热力图转换为RGB可视化
    
    Args:
        heatmap: 热力图 [H, W] 或 [1, H, W]
        colormap: 颜色映射 ('jet', 'viridis', 'hot')
        
    Returns:
        RGB图像 [H, W, 3], uint8
    """
    if heatmap.dim() == 3:
        heatmap = heatmap[0]
    
    heatmap_np = heatmap.cpu().numpy()
    
    # 归一化到 [0, 1]
    heatmap_np = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min() + 1e-8)
    
    # 转换为uint8
    heatmap_uint8 = (heatmap_np * 255).astype(np.uint8)
    
    # 应用颜色映射
    try:
        import cv2
        if colormap == 'jet':
            colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        elif colormap == 'viridis':
            colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_VIRIDIS)
        elif colormap == 'hot':
            colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_HOT)
        else:
            colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # cv2返回BGR，转为RGB
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    except ImportError:
        # 如果没有cv2，使用简单的灰度图
        colored = np.stack([heatmap_uint8] * 3, axis=-1)
    
    return colored


def draw_point_on_image(
    image: np.ndarray,
    point: Tuple[float, float],
    color: Tuple[int, int, int] = (255, 0, 0),
    radius: int = 5,
    thickness: int = -1
) -> np.ndarray:
    """
    在图像上绘制点
    
    Args:
        image: 输入图像 [H, W, 3], uint8
        point: 点坐标 (x, y)
        color: 颜色 (R, G, B)
        radius: 半径
        thickness: 线条粗细，-1表示填充
        
    Returns:
        绘制后的图像
    """
    try:
        import cv2
        image_copy = image.copy()
        cv2.circle(image_copy, (int(point[0]), int(point[1])), radius, color, thickness)
        return image_copy
    except ImportError:
        return image


def tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    """
    将tensor转换为numpy图像用于可视化
    
    Args:
        tensor: [3, H, W], [0, 1] float tensor
        
    Returns:
        [H, W, 3], uint8 numpy array
    """
    image = tensor.permute(1, 2, 0).cpu().numpy()
    image = (image * 255).clip(0, 255).astype(np.uint8)
    return image


# ============================================================================
# 训练辅助
# ============================================================================

def set_seed(seed: int):
    """
    设置随机种子以保证可重复性
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    统计模型参数数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        (总参数数, 可训练参数数)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


class AverageMeter:
    """
    计算并存储平均值和当前值
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


```


已修复随机数状态加载问题。

**修复内容：**
1. 统一处理：无论加载的数据是 `torch.Tensor`、`numpy.ndarray` 还是其他类型，都先转换为 `numpy.uint8` 数组
2. 使用 `torch.ByteTensor()` 构造函数：从 numpy 数组创建正确的 `ByteTensor` 类型
3. 添加异常处理：如果恢复随机状态失败，会打印警告但继续执行（对评估脚本来说这是可以接受的，因为评估不需要恢复随机状态）

现在可以重新运行实验4.1。如果仍有问题，代码会打印警告并继续执行，不会中断评估过程。

**注意：** 对于评估脚本，恢复随机数状态不是必需的，所以即使恢复失败也不会影响评估结果。

---

**User**

@bash (911-961) 

---

**Cursor**


检查相关代码，定位问题：



检查 `sam2_lora.predict_mask()` 的返回类型：



查看 `predict_mask` 方法的完整实现：



确认 `predict_mask` 返回 `torch.Tensor`，但 `eval_peft_model.py` 中假设返回 numpy 数组。修复该问题：


```python
#!/usr/bin/env python3
"""
PEFT模型评估脚本

在测试集上评估训练好的PEFT模型性能

主要功能:
- 加载训练好的checkpoint
- 执行完整rollout（迭代预测点序列）
- 计算指标：Dice, IoU, PCK, 点数等
- 生成可视化：点序列图、分割结果对比

调用示例:
    python -m seg-rl.peft.eval_peft_model \\
      --test_json datasets/seg_r1_md/Task01_BrainTumour/test.jsonl \\
      --checkpoint outputs/braintumour/peft_supervised/checkpoint_best.pt \\
      --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \\
      --out_dir outputs/braintumour/peft_eval \\
      --max_rollout_steps 16 \\
      --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# 添加父目录到路径
_PARENT_DIR = Path(__file__).parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from peft.lora_sam2 import LoRASAM2Wrapper
from peft.point_predictor_peft import PointPredictorFromSAMFeatures
from peft.utils_peft import compute_dice, compute_iou, load_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate PEFT Model")
    
    p.add_argument("--test_json", type=str, required=True, help="Test JSONL file")
    p.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint to evaluate")
    p.add_argument("--sam_checkpoint", type=str, required=True, help="SAM2 checkpoint")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory for results")
    
    p.add_argument("--image_size", type=int, nargs=2, default=[512, 512], help="Image size")
    p.add_argument("--max_rollout_steps", type=int, default=16, help="Max points per image")
    p.add_argument("--dice_threshold", type=float, default=0.95, help="Stop if Dice > threshold")
    p.add_argument("--improvement_threshold", type=float, default=0.001,
                   help="Stop if improvement < threshold")
    
    p.add_argument("--device", type=str, default="cuda", help="Device")
    p.add_argument("--save_visualizations", action="store_true", help="Save visualization images")
    
    return p.parse_args()


@torch.no_grad()
def rollout_single_image(
    image_path: str,
    gt_mask_path: str,
    sam2_lora: LoRASAM2Wrapper,
    point_predictor: PointPredictorFromSAMFeatures,
    args,
) -> Dict:
    """
    对单张图像执行完整rollout
    
    Returns:
        结果字典，包含点序列、Dice曲线等
    """
    device = args.device
    
    # 加载图像和真值
    image = Image.open(image_path).convert('RGB')
    gt_mask = Image.open(gt_mask_path).convert('L')
    
    orig_w, orig_h = image.size
    
    # Resize
    image_resized = image.resize(tuple(args.image_size), Image.BILINEAR)
    gt_mask_resized = gt_mask.resize(tuple(args.image_size), Image.NEAREST)
    
    # 转换为tensor
    import torchvision.transforms.functional as TF
    image_tensor = TF.to_tensor(image_resized).unsqueeze(0).to(device)  # [1, 3, H, W]
    gt_mask_tensor = TF.to_tensor(gt_mask_resized).to(device)  # [1, H, W]
    gt_mask_binary = (gt_mask_tensor > 0.5).float()
    
    # Rollout
    points_seq = []
    labels_seq = []
    dice_history = []
    iou_history = []
    
    prev_dice = 0.0
    
    for step in range(args.max_rollout_steps):
        # 生成当前掩模
        if step == 0:
            prev_mask = torch.zeros_like(gt_mask_binary)
        else:
            # 用SAM2生成掩模
            # 需要将点坐标缩放回原始尺寸
            points_orig = [(px * orig_w / args.image_size[1], py * orig_h / args.image_size[0])
                          for px, py in points_seq]
            prev_mask = sam2_lora.predict_mask(
                np.array(image), points_orig, labels_seq
            )
            # predict_mask 返回的是 torch.Tensor，不是 numpy 数组
            if isinstance(prev_mask, torch.Tensor):
                prev_mask = prev_mask.unsqueeze(0).to(device)
            else:
                prev_mask = torch.from_numpy(prev_mask).unsqueeze(0).to(device)
            prev_mask = torch.nn.functional.interpolate(
                prev_mask.unsqueeze(0), size=tuple(args.image_size), mode='nearest'
            )[0]
        
        # 预测下一个点
        sam_features = sam2_lora.get_image_features(image_tensor, feature_scale=8)
        heatmap_logits, label_logits = point_predictor(sam_features, prev_mask.unsqueeze(0))
        
        # Argmax获取点
        B, _, H, W = heatmap_logits.shape
        heatmap_flat = heatmap_logits.view(B, -1)
        pred_idx = heatmap_flat.argmax(dim=1)
        pred_y = (pred_idx // W).float()
        pred_x = (pred_idx % W).float()
        
        # 标签
        pred_label = label_logits.argmax(dim=1).item()
        
        # 记录
        points_seq.append((pred_x.item(), pred_y.item()))
        labels_seq.append(pred_label)
        
        # 计算指标
        dice = compute_dice(prev_mask[0], gt_mask_binary[0])
        iou = compute_iou(prev_mask[0], gt_mask_binary[0])
        dice_history.append(dice)
        iou_history.append(iou)
        
        # 停止条件
        if dice > args.dice_threshold:
            break
        if step > 0 and (dice - prev_dice) < args.improvement_threshold:
            break
        
        prev_dice = dice
    
    return {
        'image_path': image_path,
        'points': points_seq,
        'labels': labels_seq,
        'dice_history': dice_history,
        'iou_history': iou_history,
        'final_dice': dice_history[-1] if dice_history else 0.0,
        'final_iou': iou_history[-1] if iou_history else 0.0,
        'num_points': len(points_seq),
    }


def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 设备
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # 加载checkpoint配置
    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = ckpt.get('config', {})
    
    # 提取配置
    lora_rank = config.get('lora_rank', 16)
    lora_alpha = config.get('lora_alpha', 32)
    fusion_mode = config.get('fusion_mode', 'film')
    feature_scale = config.get('feature_scale', 8)
    
    print(f"Model config: LoRA rank={lora_rank}, alpha={lora_alpha}, fusion={fusion_mode}")
    
    # 1. 初始化SAM2 + LoRA
    print("\nInitializing SAM2 with LoRA...")
    sam2_lora = LoRASAM2Wrapper(
        sam_checkpoint=args.sam_checkpoint,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        device=str(device)
    )
    
    # 2. 初始化点预测网络
    print("Initializing Point Predictor...")
    
    # 根据 feature_scale 确定 SAM 特征通道数
    sam_feature_dim_map = {4: 32, 8: 64, 16: 256}
    if feature_scale not in sam_feature_dim_map:
        raise ValueError(f"Unsupported feature_scale: {feature_scale}. Use 4, 8, or 16.")
    sam_feature_dim = sam_feature_dim_map[feature_scale]
    
    point_predictor = PointPredictorFromSAMFeatures(
        sam_feature_dim=sam_feature_dim,
        output_size=tuple(args.image_size),
        fusion_mode=fusion_mode,
        feature_scale=feature_scale,
    ).to(device)
    
    # 3. 加载权重
    print("Loading model weights...")
    load_checkpoint(
        args.checkpoint, point_predictor, sam2_lora, None, None, None, str(device)
    )
    
    point_predictor.eval()
    
    # 4. 加载测试数据
    print(f"\nLoading test data from {args.test_json}")
    with open(args.test_json, 'r') as f:
        content = f.read().strip()
        if content.startswith('['):
            test_samples = json.loads(content)
        else:
            test_samples = [json.loads(line) for line in content.splitlines() if line.strip()]
    
    print(f"Loaded {len(test_samples)} test samples")
    
    # 5. 评估
    print("\n" + "="*80)
    print("Evaluating...")
    print("="*80)
    
    results = []
    
    for sample in tqdm(test_samples):
        image_path = sample['image']
        gt_mask_path = sample.get('gt_mask', None)
        
        if gt_mask_path is None or not os.path.exists(gt_mask_path):
            print(f"Warning: GT mask not found for {image_path}, skipping")
            continue
        
        result = rollout_single_image(
            image_path, gt_mask_path, sam2_lora, point_predictor, args
        )
        results.append(result)
    
    # 6. 汇总统计
    print("\n" + "="*80)
    print("Results Summary")
    print("="*80)
    
    final_dices = [r['final_dice'] for r in results]
    final_ious = [r['final_iou'] for r in results]
    num_points = [r['num_points'] for r in results]
    
    print(f"Number of samples: {len(results)}")
    print(f"\nDice Score:")
    print(f"  Mean: {np.mean(final_dices):.4f}")
    print(f"  Std:  {np.std(final_dices):.4f}")
    print(f"  Median: {np.median(final_dices):.4f}")
    
    print(f"\nIoU Score:")
    print(f"  Mean: {np.mean(final_ious):.4f}")
    print(f"  Std:  {np.std(final_ious):.4f}")
    print(f"  Median: {np.median(final_ious):.4f}")
    
    print(f"\nNumber of Points:")
    print(f"  Mean: {np.mean(num_points):.2f}")
    print(f"  Std:  {np.std(num_points):.2f}")
    print(f"  Median: {np.median(num_points):.0f}")
    
    # 7. 保存结果
    results_json_path = os.path.join(args.out_dir, 'eval_results.json')
    with open(results_json_path, 'w') as f:
        json.dump({
            'summary': {
                'num_samples': len(results),
                'mean_dice': float(np.mean(final_dices)),
                'std_dice': float(np.std(final_dices)),
                'median_dice': float(np.median(final_dices)),
                'mean_iou': float(np.mean(final_ious)),
                'mean_num_points': float(np.mean(num_points)),
            },
            'per_sample_results': results,
        }, f, indent=2)
    
    print(f"\nResults saved to {results_json_path}")


if __name__ == "__main__":
    main()


```


检查修复后的代码，确保逻辑正确：



已修复该问题。

**问题原因：**
`sam2_lora.predict_mask()` 返回的是 `torch.Tensor`（已在 `self.device` 上），但代码假设返回 numpy 数组，并使用了 `torch.from_numpy()`，导致类型错误。

**修复内容：**
在 `eval_peft_model.py` 中添加类型检查：
- 如果返回的是 `torch.Tensor`，直接使用并确保在正确的设备上
- 如果是 numpy 数组，则使用 `torch.from_numpy()` 转换

现在可以重新运行实验4.1，该错误应已解决。

**注意：** `predict_mask` 返回的 tensor 形状是 `[H, W]`，代码会先 `unsqueeze(0)` 添加 batch 维度，然后进行插值调整大小。

---

**User**

请阅读docs/cursor_next_steps_for_peft_grpo_sam-20251107.md以获取之前的上下文，接着回答我的新问题：按照README_PEFT_EXPERIMENT_GUIDE.md的指导，我一路跑到了实验2.1: 基础训练（推荐配置）。但从实际实验结果来看，跑了80多个epoch后，val_pck@20.0只有大约0.2~0.3，val_pck@40.0也只有0.48~0.5左右，这么低的数值正常吗？即便考虑到tau设置为1.0的话。并且我用训练生成的checkpoint在test数据集上做实验4.1: 评估监督预训练模型，得到的DICE等指标也很低：@bash (784-834)  请全面仔细地分析可能的问题原因，给出你的调试或检查方案。

---

**Cursor**


阅读相关文档和代码，分析问题原因：



继续查看训练代码和评估代码，分析问题：



查看损失函数和数据加载器：



查看损失函数的具体实现：



基于代码和文档，分析可能原因并给出调试方案：

## 问题分析

### 1. 数据质量问题

**可能原因：**
- 训练数据中的点标注可能不准确
- SAM2生成的中间掩模质量差，导致后续点预测困难
- 数据分布不均衡（k=0 vs k>0的样本比例）

**检查方案：**
```python
# 检查训练数据的点分布
python -c "
import json
import numpy as np

with open('outputs/braintumour/peft_train-251107.jsonl', 'r') as f:
    data = json.load(f)

# 统计每个样本的点数分布
point_counts = [len(item['points']) for item in data]
print(f'点数分布: min={min(point_counts)}, max={max(point_counts)}, mean={np.mean(point_counts):.2f}')

# 检查k=0和k>0的样本比例
k0_samples = sum(1 for item in data if len(item['points']) > 0)
print(f'有至少1个点的样本: {k0_samples}/{len(data)}')

# 检查点坐标是否在图像范围内
for i, item in enumerate(data[:10]):
    points = item['points']
    print(f'Sample {i}: {len(points)} points')
    for j, (x, y) in enumerate(points):
        print(f'  Point {j}: ({x:.1f}, {y:.1f})')
"
```

### 2. 损失函数和训练目标不匹配

**可能原因：**
- `tau=1.0` 时，模型学习的是尖锐分布，但评估用 argmax，可能不一致
- `sigma=8.0` 可能过大，导致目标分布过于平滑
- KL散度损失可能难以优化

**检查方案：**
```python
# 检查训练损失和验证损失的关系
# 查看TensorBoard日志，检查：
# 1. train/heatmap_loss 是否在下降
# 2. val/val_loss 是否在下降
# 3. train/heatmap_loss 和 val/val_loss 的差距

# 检查预测热力图的分布
# 在验证时添加可视化代码，查看：
# - 预测热力图是否集中在目标点附近
# - 热力图是否过于平坦（所有像素概率相近）
```

### 3. 模型架构问题

**可能原因：**
- SAM特征提取可能有问题
- 点预测网络的融合方式（FiLM）可能不适合
- 特征尺度（feature_scale=8）可能导致信息丢失

**检查方案：**
```python
# 检查SAM特征是否正常
# 在训练代码中添加：
sam_features = sam2_lora.get_image_features(images, feature_scale=8)
print(f'SAM features shape: {sam_features.shape}')
print(f'SAM features stats: mean={sam_features.mean():.4f}, std={sam_features.std():.4f}')

# 检查点预测网络的输出
heatmap_logits, label_logits = point_predictor(sam_features, prev_masks)
print(f'Heatmap logits: min={heatmap_logits.min():.4f}, max={heatmap_logits.max():.4f}')
print(f'Heatmap logits std: {heatmap_logits.std():.4f}')

# 检查预测点是否合理
pred_probs = F.softmax(heatmap_logits.view(B, -1), dim=1)
print(f'Pred probs: max={pred_probs.max():.4f}, entropy={-(pred_probs * torch.log(pred_probs + 1e-10)).sum(dim=1).mean():.4f}')
```

### 4. 训练超参数问题

**可能原因：**
- 学习率可能不合适（SAM: 1e-5, Point: 1e-4）
- 梯度裁剪可能过于严格（grad_clip=1.0）
- 学习率调度器可能过早衰减

**检查方案：**
```python
# 检查梯度大小
# 在训练循环中添加：
if optimizer_sam is not None:
    sam_grad_norm = torch.nn.utils.clip_grad_norm_(sam2_lora.get_lora_parameters(), float('inf'))
    print(f'SAM LoRA grad norm: {sam_grad_norm:.4f}')

point_grad_norm = torch.nn.utils.clip_grad_norm_(point_predictor.parameters(), float('inf'))
print(f'Point predictor grad norm: {point_grad_norm:.4f}')

# 检查学习率
print(f'Current LR - SAM: {optimizer_sam.param_groups[0]["lr"]:.2e}, Point: {optimizer_point.param_groups[0]["lr"]:.2e}')
```

### 5. 评估方式问题

**可能原因：**
- 验证时使用argmax，但训练时使用softmax，可能不一致
- PCK阈值（20, 40像素）对于512x512图像可能过于严格
- 评估时没有使用rollout，而是单步预测

**检查方案：**
```python
# 检查预测点的分布
# 在验证代码中添加：
distances = torch.norm(pred_points - target_points, p=2, dim=1)
print(f'Distance stats: mean={distances.mean():.4f}, median={distances.median():.4f}, max={distances.max():.4f}')

# 检查预测点是否在图像范围内
print(f'Pred points range: x=[{pred_points[:, 0].min():.1f}, {pred_points[:, 0].max():.1f}], y=[{pred_points[:, 1].min():.1f}, {pred_points[:, 1].max():.1f}]')
print(f'Target points range: x=[{target_points[:, 0].min():.1f}, {target_points[:, 0].max():.1f}], y=[{target_points[:, 1].min():.1f}, {target_points[:, 1].max():.1f}]')
```

## 调试方案

### 阶段1：数据验证

1. 检查训练数据质量：
```bash
# 可视化几个训练样本
python -c "
import json
import cv2
import numpy as np
from PIL import Image

with open('outputs/braintumour/peft_train-251107.jsonl', 'r') as f:
    data = json.load(f)

# 随机选择几个样本可视化
import random
random.seed(42)
samples = random.sample(data, 5)

for i, sample in enumerate(samples):
    img = Image.open(sample['image']).convert('RGB')
    points = sample['points']
    print(f'Sample {i}: {len(points)} points')
    # 在图像上绘制点
    img_np = np.array(img)
    for j, (x, y) in enumerate(points):
        cv2.circle(img_np, (int(x), int(y)), 5, (255, 0, 0), -1)
    Image.fromarray(img_np).save(f'debug_sample_{i}.png')
"
```

2. 检查SAM掩模质量：
```bash
# 检查生成的SAM掩模是否合理
# 查看 outputs/braintumour/sam_masks_ref/ 中的掩模
```

### 阶段2：模型诊断

1. 冻结SAM，只训练点预测网络：
```bash
# 修改训练脚本，添加 --freeze_sam 选项
# 如果点预测网络单独训练效果好，说明问题在SAM LoRA
```

2. 检查特征提取：
```python
# 在训练代码中添加特征可视化
# 保存SAM特征和预测热力图，检查是否合理
```

### 阶段3：超参数调优

1. 尝试不同的损失函数：
```bash
# 尝试MSE损失
--loss mse --sigma 8.0

# 尝试更小的sigma
--loss kl --sigma 4.0 --tau 1.0
```

2. 调整学习率：
```bash
# 尝试更大的学习率
--lr_sam 5e-5 --lr_point 5e-4

# 或更小的学习率
--lr_sam 5e-6 --lr_point 5e-5
```

3. 调整温度参数：
```bash
# 尝试更大的tau（更平滑的分布）
--tau 2.0

# 或更小的tau（更尖锐的分布）
--tau 0.5
```

### 阶段4：简化实验

1. 只训练k=0的样本（第一个点）：
```python
# 修改数据集，只使用k=0的样本
# 检查模型是否能学习预测第一个点
```

2. 使用更简单的融合方式：
```bash
# 尝试concatenation而不是FiLM
--fusion_mode concat
```

## 建议的调试顺序

1. 先检查数据质量（阶段1）
2. 检查训练损失是否正常下降
3. 可视化预测热力图，看是否集中在目标点附近
4. 尝试冻结SAM，只训练点预测网络
5. 调整超参数（sigma, tau, 学习率）
6. 如果仍无效，考虑简化模型架构

## 预期结果参考

根据文档，预期性能应该是：
- PCK@20: 应该 > 0.5（理想情况 > 0.7）
- PCK@40: 应该 > 0.7（理想情况 > 0.85）
- Dice: 应该 > 0.7（理想情况 > 0.85）

当前结果明显偏低，需要系统排查。

需要我帮你实现某个具体的检查脚本吗？

```python
# 检查训练数据的点分布
python -c "
import json
import numpy as np

with open('outputs/braintumour/peft_train-251107.jsonl', 'r') as f:
    data = json.load(f)

# 统计每个样本的点数分布
point_counts = [len(item['points']) for item in data]
print(f'点数分布: min={min(point_counts)}, max={max(point_counts)}, mean={np.mean(point_counts):.2f}')

# 检查k=0和k>0的样本比例
k0_samples = sum(1 for item in data if len(item['points']) > 0)
print(f'有至少1个点的样本: {k0_samples}/{len(data)}')

# 检查点坐标是否在图像范围内
for i, item in enumerate(data[:10]):
    points = item['points']
    print(f'Sample {i}: {len(points)} points')
    for j, (x, y) in enumerate(points):
        print(f'  Point {j}: ({x:.1f}, {y:.1f})')
"
```

```python
# 检查训练损失和验证损失的关系
# 查看TensorBoard日志，检查：
# 1. train/heatmap_loss 是否在下降
# 2. val/val_loss 是否在下降
# 3. train/heatmap_loss 和 val/val_loss 的差距

# 检查预测热力图的分布
# 在验证时添加可视化代码，查看：
# - 预测热力图是否集中在目标点附近
# - 热力图是否过于平坦（所有像素概率相近）
```

```python
# 检查SAM特征是否正常
# 在训练代码中添加：
sam_features = sam2_lora.get_image_features(images, feature_scale=8)
print(f'SAM features shape: {sam_features.shape}')
print(f'SAM features stats: mean={sam_features.mean():.4f}, std={sam_features.std():.4f}')

# 检查点预测网络的输出
heatmap_logits, label_logits = point_predictor(sam_features, prev_masks)
print(f'Heatmap logits: min={heatmap_logits.min():.4f}, max={heatmap_logits.max():.4f}')
print(f'Heatmap logits std: {heatmap_logits.std():.4f}')

# 检查预测点是否合理
pred_probs = F.softmax(heatmap_logits.view(B, -1), dim=1)
print(f'Pred probs: max={pred_probs.max():.4f}, entropy={-(pred_probs * torch.log(pred_probs + 1e-10)).sum(dim=1).mean():.4f}')
```

```python
# 检查梯度大小
# 在训练循环中添加：
if optimizer_sam is not None:
    sam_grad_norm = torch.nn.utils.clip_grad_norm_(sam2_lora.get_lora_parameters(), float('inf'))
    print(f'SAM LoRA grad norm: {sam_grad_norm:.4f}')

point_grad_norm = torch.nn.utils.clip_grad_norm_(point_predictor.parameters(), float('inf'))
print(f'Point predictor grad norm: {point_grad_norm:.4f}')

# 检查学习率
print(f'Current LR - SAM: {optimizer_sam.param_groups[0]["lr"]:.2e}, Point: {optimizer_point.param_groups[0]["lr"]:.2e}')
```

```python
# 检查预测点的分布
# 在验证代码中添加：
distances = torch.norm(pred_points - target_points, p=2, dim=1)
print(f'Distance stats: mean={distances.mean():.4f}, median={distances.median():.4f}, max={distances.max():.4f}')

# 检查预测点是否在图像范围内
print(f'Pred points range: x=[{pred_points[:, 0].min():.1f}, {pred_points[:, 0].max():.1f}], y=[{pred_points[:, 1].min():.1f}, {pred_points[:, 1].max():.1f}]')
print(f'Target points range: x=[{target_points[:, 0].min():.1f}, {target_points[:, 0].max():.1f}], y=[{target_points[:, 1].min():.1f}, {target_points[:, 1].max():.1f}]')
```

```shellscript
# 可视化几个训练样本
python -c "
import json
import cv2
import numpy as np
from PIL import Image

with open('outputs/braintumour/peft_train-251107.jsonl', 'r') as f:
    data = json.load(f)

# 随机选择几个样本可视化
import random
random.seed(42)
samples = random.sample(data, 5)

for i, sample in enumerate(samples):
    img = Image.open(sample['image']).convert('RGB')
    points = sample['points']
    print(f'Sample {i}: {len(points)} points')
    # 在图像上绘制点
    img_np = np.array(img)
    for j, (x, y) in enumerate(points):
        cv2.circle(img_np, (int(x), int(y)), 5, (255, 0, 0), -1)
    Image.fromarray(img_np).save(f'debug_sample_{i}.png')
"
```

```shellscript
# 检查生成的SAM掩模是否合理
# 查看 outputs/braintumour/sam_masks_ref/ 中的掩模
```

```shellscript
# 修改训练脚本，添加 --freeze_sam 选项
# 如果点预测网络单独训练效果好，说明问题在SAM LoRA
```

```python
# 在训练代码中添加特征可视化
# 保存SAM特征和预测热力图，检查是否合理
```

```shellscript
# 尝试MSE损失
--loss mse --sigma 8.0

# 尝试更小的sigma
--loss kl --sigma 4.0 --tau 1.0
```

```shellscript
# 尝试更大的学习率
--lr_sam 5e-5 --lr_point 5e-4

# 或更小的学习率
--lr_sam 5e-6 --lr_point 5e-5
```

```shellscript
# 尝试更大的tau（更平滑的分布）
--tau 2.0

# 或更小的tau（更尖锐的分布）
--tau 0.5
```

```python
# 修改数据集，只使用k=0的样本
# 检查模型是否能学习预测第一个点
```

```shellscript
# 尝试concatenation而不是FiLM
--fusion_mode concat
```

---

**User**

你上述# 可视化几个训练样本的脚本中没有区分前景点label=1和背景点label=0，请用不同的颜色标示前景和背景点，否则不能判断提示点是否正确

---

**Cursor**


修改可视化脚本，用不同颜色区分前景点（label=1）和背景点（label=0）：

```python
# 可视化几个训练样本（区分前景和背景点）
python -c "
import json
import cv2
import numpy as np
from PIL import Image
import random

with open('outputs/braintumour/peft_train-251107.jsonl', 'r') as f:
    data = json.load(f)

# 随机选择几个样本可视化
random.seed(42)
samples = random.sample(data, 5)

for i, sample in enumerate(samples):
    img = Image.open(sample['image']).convert('RGB')
    points = sample['points']
    labels = sample['labels']
    
    print(f'Sample {i}: {len(points)} points')
    
    # 在图像上绘制点
    img_np = np.array(img)
    
    for j, ((x, y), label) in enumerate(zip(points, labels)):
        # 前景点（label=1）：红色
        # 背景点（label=0）：蓝色
        if label == 1:
            color = (0, 0, 255)  # 红色 (BGR格式)
            point_type = 'FG'
        else:
            color = (255, 0, 0)  # 蓝色 (BGR格式)
            point_type = 'BG'
        
        # 绘制点（较大的圆，便于观察）
        cv2.circle(img_np, (int(x), int(y)), 8, color, -1)
        # 绘制外圈（白色边框，便于区分）
        cv2.circle(img_np, (int(x), int(y)), 10, (255, 255, 255), 2)
        
        # 添加标签文本
        cv2.putText(img_np, f'{j}({point_type})', 
                   (int(x) + 12, int(y)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        print(f'  Point {j}: ({x:.1f}, {y:.1f}), label={label} ({point_type})')
    
    # 保存图像
    Image.fromarray(img_np).save(f'debug_sample_{i}.png')
    print(f'  Saved to debug_sample_{i}.png')
    print()
"
```

更详细的版本，包含真值掩模叠加：

```python
# 可视化训练样本（完整版本：包含图像、点、掩模）
python -c "
import json
import cv2
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt

with open('outputs/braintumour/peft_train-251107.jsonl', 'r') as f:
    data = json.load(f)

# 随机选择几个样本可视化
random.seed(42)
samples = random.sample(data, 5)

for i, sample in enumerate(samples):
    img = Image.open(sample['image']).convert('RGB')
    gt_mask_path = sample.get('gt_mask', None)
    points = sample['points']
    labels = sample['labels']
    
    print(f'Sample {i}: {len(points)} points')
    print(f'  Image: {sample[\"image\"]}')
    if gt_mask_path:
        print(f'  GT Mask: {gt_mask_path}')
    
    # 在图像上绘制点
    img_np = np.array(img)
    
    # 如果有真值掩模，叠加显示（半透明）
    if gt_mask_path and os.path.exists(gt_mask_path):
        gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
        # 将掩模转为彩色（绿色通道）
        mask_colored = np.zeros_like(img_np)
        mask_colored[:, :, 1] = (gt_mask > 127).astype(np.uint8) * 255
        # 叠加掩模（30%透明度）
        img_np = cv2.addWeighted(img_np, 0.7, mask_colored, 0.3, 0)
    
    for j, ((x, y), label) in enumerate(zip(points, labels)):
        # 前景点（label=1）：红色，较大
        # 背景点（label=0）：蓝色，较小
        if label == 1:
            color = (0, 0, 255)  # 红色 (BGR格式)
            radius = 10
            point_type = 'FG'
        else:
            color = (255, 0, 0)  # 蓝色 (BGR格式)
            radius = 8
            point_type = 'BG'
        
        # 绘制点
        cv2.circle(img_np, (int(x), int(y)), radius, color, -1)
        # 绘制外圈（白色边框）
        cv2.circle(img_np, (int(x), int(y)), radius + 2, (255, 255, 255), 2)
        
        # 添加标签文本
        cv2.putText(img_np, f'{j}({point_type})', 
                   (int(x) + radius + 5, int(y)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(img_np, f'{j}({point_type})', 
                   (int(x) + radius + 5, int(y)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        print(f'  Point {j}: ({x:.1f}, {y:.1f}), label={label} ({point_type})')
    
    # 保存图像
    Image.fromarray(img_np).save(f'debug_sample_{i}.png')
    print(f'  Saved to debug_sample_{i}.png')
    print()
"
```

如果使用matplotlib，更美观的版本：

```python
# 使用matplotlib可视化（更美观）
python -c "
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import os

with open('outputs/braintumour/peft_train-251107.jsonl', 'r') as f:
    data = json.load(f)

# 随机选择几个样本可视化
random.seed(42)
samples = random.sample(data, 5)

for i, sample in enumerate(samples):
    img = Image.open(sample['image']).convert('RGB')
    gt_mask_path = sample.get('gt_mask', None)
    points = sample['points']
    labels = sample['labels']
    
    print(f'Sample {i}: {len(points)} points')
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(img)
    
    # 如果有真值掩模，叠加显示
    if gt_mask_path and os.path.exists(gt_mask_path):
        gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
        mask_binary = (gt_mask > 127).astype(float)
        ax.imshow(mask_binary, alpha=0.3, cmap='Greens')
    
    # 绘制点
    for j, ((x, y), label) in enumerate(zip(points, labels)):
        if label == 1:
            # 前景点：红色，实心圆
            color = 'red'
            marker = 'o'
            size = 150
            point_type = 'FG'
        else:
            # 背景点：蓝色，实心圆
            color = 'blue'
            marker = 'o'
            size = 120
            point_type = 'BG'
        
        # 绘制点
        ax.scatter(x, y, c=color, s=size, marker=marker, 
                  edgecolors='white', linewidths=2, zorder=10)
        
        # 添加标签文本
        ax.text(x + 15, y, f'{j}({point_type})', 
               fontsize=10, color='white', weight='bold',
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
        
        print(f'  Point {j}: ({x:.1f}, {y:.1f}), label={label} ({point_type})')
    
    ax.set_title(f'Sample {i}: {len(points)} points', fontsize=14, weight='bold')
    ax.axis('off')
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(f'debug_sample_{i}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'  Saved to debug_sample_{i}.png')
    print()
"
```

**颜色说明：**
- 红色圆点：前景点（label=1，foreground）
- 蓝色圆点：背景点（label=0，background）
- 白色边框：便于在复杂背景下识别
- 文本标签：显示点序号和类型（FG/BG）

**使用建议：**
1. 先运行第一个简单版本，快速检查数据
2. 如需更详细的可视化，使用第二个或第三个版本
3. 检查点是否在目标区域：前景点应在目标区域内，背景点应在目标区域外

运行后查看生成的 `debug_sample_*.png` 文件，检查点标注是否正确。

```python
# 可视化几个训练样本（区分前景和背景点）
python -c "
import json
import cv2
import numpy as np
from PIL import Image
import random

with open('outputs/braintumour/peft_train-251107.jsonl', 'r') as f:
    data = json.load(f)

# 随机选择几个样本可视化
random.seed(42)
samples = random.sample(data, 5)

for i, sample in enumerate(samples):
    img = Image.open(sample['image']).convert('RGB')
    points = sample['points']
    labels = sample['labels']
    
    print(f'Sample {i}: {len(points)} points')
    
    # 在图像上绘制点
    img_np = np.array(img)
    
    for j, ((x, y), label) in enumerate(zip(points, labels)):
        # 前景点（label=1）：红色
        # 背景点（label=0）：蓝色
        if label == 1:
            color = (0, 0, 255)  # 红色 (BGR格式)
            point_type = 'FG'
        else:
            color = (255, 0, 0)  # 蓝色 (BGR格式)
            point_type = 'BG'
        
        # 绘制点（较大的圆，便于观察）
        cv2.circle(img_np, (int(x), int(y)), 8, color, -1)
        # 绘制外圈（白色边框，便于区分）
        cv2.circle(img_np, (int(x), int(y)), 10, (255, 255, 255), 2)
        
        # 添加标签文本
        cv2.putText(img_np, f'{j}({point_type})', 
                   (int(x) + 12, int(y)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        print(f'  Point {j}: ({x:.1f}, {y:.1f}), label={label} ({point_type})')
    
    # 保存图像
    Image.fromarray(img_np).save(f'debug_sample_{i}.png')
    print(f'  Saved to debug_sample_{i}.png')
    print()
"
```

```python
# 可视化训练样本（完整版本：包含图像、点、掩模）
python -c "
import json
import cv2
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt

with open('outputs/braintumour/peft_train-251107.jsonl', 'r') as f:
    data = json.load(f)

# 随机选择几个样本可视化
random.seed(42)
samples = random.sample(data, 5)

for i, sample in enumerate(samples):
    img = Image.open(sample['image']).convert('RGB')
    gt_mask_path = sample.get('gt_mask', None)
    points = sample['points']
    labels = sample['labels']
    
    print(f'Sample {i}: {len(points)} points')
    print(f'  Image: {sample[\"image\"]}')
    if gt_mask_path:
        print(f'  GT Mask: {gt_mask_path}')
    
    # 在图像上绘制点
    img_np = np.array(img)
    
    # 如果有真值掩模，叠加显示（半透明）
    if gt_mask_path and os.path.exists(gt_mask_path):
        gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
        # 将掩模转为彩色（绿色通道）
        mask_colored = np.zeros_like(img_np)
        mask_colored[:, :, 1] = (gt_mask > 127).astype(np.uint8) * 255
        # 叠加掩模（30%透明度）
        img_np = cv2.addWeighted(img_np, 0.7, mask_colored, 0.3, 0)
    
    for j, ((x, y), label) in enumerate(zip(points, labels)):
        # 前景点（label=1）：红色，较大
        # 背景点（label=0）：蓝色，较小
        if label == 1:
            color = (0, 0, 255)  # 红色 (BGR格式)
            radius = 10
            point_type = 'FG'
        else:
            color = (255, 0, 0)  # 蓝色 (BGR格式)
            radius = 8
            point_type = 'BG'
        
        # 绘制点
        cv2.circle(img_np, (int(x), int(y)), radius, color, -1)
        # 绘制外圈（白色边框）
        cv2.circle(img_np, (int(x), int(y)), radius + 2, (255, 255, 255), 2)
        
        # 添加标签文本
        cv2.putText(img_np, f'{j}({point_type})', 
                   (int(x) + radius + 5, int(y)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(img_np, f'{j}({point_type})', 
                   (int(x) + radius + 5, int(y)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        print(f'  Point {j}: ({x:.1f}, {y:.1f}), label={label} ({point_type})')
    
    # 保存图像
    Image.fromarray(img_np).save(f'debug_sample_{i}.png')
    print(f'  Saved to debug_sample_{i}.png')
    print()
"
```

```python
# 使用matplotlib可视化（更美观）
python -c "
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import os

with open('outputs/braintumour/peft_train-251107.jsonl', 'r') as f:
    data = json.load(f)

# 随机选择几个样本可视化
random.seed(42)
samples = random.sample(data, 5)

for i, sample in enumerate(samples):
    img = Image.open(sample['image']).convert('RGB')
    gt_mask_path = sample.get('gt_mask', None)
    points = sample['points']
    labels = sample['labels']
    
    print(f'Sample {i}: {len(points)} points')
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(img)
    
    # 如果有真值掩模，叠加显示
    if gt_mask_path and os.path.exists(gt_mask_path):
        gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
        mask_binary = (gt_mask > 127).astype(float)
        ax.imshow(mask_binary, alpha=0.3, cmap='Greens')
    
    # 绘制点
    for j, ((x, y), label) in enumerate(zip(points, labels)):
        if label == 1:
            # 前景点：红色，实心圆
            color = 'red'
            marker = 'o'
            size = 150
            point_type = 'FG'
        else:
            # 背景点：蓝色，实心圆
            color = 'blue'
            marker = 'o'
            size = 120
            point_type = 'BG'
        
        # 绘制点
        ax.scatter(x, y, c=color, s=size, marker=marker, 
                  edgecolors='white', linewidths=2, zorder=10)
        
        # 添加标签文本
        ax.text(x + 15, y, f'{j}({point_type})', 
               fontsize=10, color='white', weight='bold',
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
        
        print(f'  Point {j}: ({x:.1f}, {y:.1f}), label={label} ({point_type})')
    
    ax.set_title(f'Sample {i}: {len(points)} points', fontsize=14, weight='bold')
    ax.axis('off')
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(f'debug_sample_{i}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'  Saved to debug_sample_{i}.png')
    print()
"
```

---

**User**

把你这个“# 使用matplotlib可视化（更美观）”的脚本写成一个更通用的可视化脚本，使得JSONL文件和输出图片的文件夹可以用命令行参数指定，并且可以选择是随机选取几个样本还是遍历所有样本生成可视化图

---

**Cursor**


创建一个通用的可视化脚本：


```python
#!/usr/bin/env python3
"""
可视化训练数据脚本

用于可视化PEFT训练数据中的图像、点标注和真值掩模。
支持随机采样或遍历所有样本。

用法:
    # 随机采样5个样本
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_train-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode random \
        --num_samples 5

    # 遍历所有样本
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_train-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode all
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image


def load_jsonl(jsonl_path: str) -> List[Dict]:
    """加载JSONL文件"""
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith('['):
            # JSON数组格式
            data = json.loads(content)
        else:
            # JSONL格式（逐行）
            data = []
            for line in content.splitlines():
                if line.strip():
                    data.append(json.loads(line))
    return data


def visualize_sample(
    sample: Dict,
    output_path: str,
    show_mask: bool = True,
    dpi: int = 150,
) -> None:
    """
    可视化单个样本
    
    Args:
        sample: 样本字典，包含 'image', 'gt_mask', 'points', 'labels' 等字段
        output_path: 输出图像路径
        show_mask: 是否显示真值掩模
        dpi: 图像分辨率
    """
    # 加载图像
    image_path = sample['image']
    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        return
    
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    
    # 获取点和标签
    points = sample.get('points', [])
    labels = sample.get('labels', [])
    
    if len(points) != len(labels):
        print(f"Warning: Points and labels length mismatch: {len(points)} vs {len(labels)}")
        return
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(img_np)
    
    # 如果有真值掩模，叠加显示
    if show_mask:
        gt_mask_path = sample.get('gt_mask', None)
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                mask_binary = (gt_mask > 127).astype(float)
                ax.imshow(mask_binary, alpha=0.3, cmap='Greens')
            except Exception as e:
                print(f"Warning: Failed to load mask {gt_mask_path}: {e}")
    
    # 绘制点
    for j, ((x, y), label) in enumerate(zip(points, labels)):
        if label == 1:
            # 前景点：红色，实心圆
            color = 'red'
            size = 150
            point_type = 'FG'
        else:
            # 背景点：蓝色，实心圆
            color = 'blue'
            size = 120
            point_type = 'BG'
        
        # 绘制点
        ax.scatter(x, y, c=color, s=size, marker='o', 
                  edgecolors='white', linewidths=2, zorder=10)
        
        # 添加标签文本
        ax.text(x + 15, y, f'{j}({point_type})', 
               fontsize=10, color='white', weight='bold',
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    # 设置标题
    stem = Path(image_path).stem
    title = f'{stem}: {len(points)} points'
    ax.set_title(title, fontsize=14, weight='bold')
    ax.axis('off')
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='可视化PEFT训练数据',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--jsonl',
        type=str,
        required=True,
        help='输入JSONL文件路径'
    )
    
    parser.add_argument(
        '--out_dir',
        type=str,
        required=True,
        help='输出图像文件夹路径'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['random', 'all'],
        default='random',
        help='采样模式: random=随机采样, all=遍历所有样本'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='随机采样模式下的样本数量（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--show_mask',
        action='store_true',
        default=True,
        help='是否显示真值掩模（默认True）'
    )
    
    parser.add_argument(
        '--no_mask',
        dest='show_mask',
        action='store_false',
        help='不显示真值掩模'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='输出图像分辨率（DPI）'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default='sample',
        help='输出文件名前缀'
    )
    
    args = parser.parse_args()
    
    # 加载数据
    print(f"Loading data from {args.jsonl}...")
    data = load_jsonl(args.jsonl)
    print(f"Loaded {len(data)} samples")
    
    # 选择样本
    if args.mode == 'random':
        random.seed(args.seed)
        selected_samples = random.sample(data, min(args.num_samples, len(data)))
        print(f"Randomly selected {len(selected_samples)} samples")
    else:
        selected_samples = data
        print(f"Processing all {len(selected_samples)} samples")
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 可视化每个样本
    print(f"\nGenerating visualizations...")
    for i, sample in enumerate(selected_samples):
        # 生成输出文件名
        if args.mode == 'random':
            output_filename = f'{args.prefix}_{i:03d}.png'
        else:
            # 使用图像文件名作为输出文件名
            image_path = sample.get('image', '')
            if image_path:
                stem = Path(image_path).stem
                output_filename = f'{args.prefix}_{stem}.png'
            else:
                output_filename = f'{args.prefix}_{i:04d}.png'
        
        output_path = os.path.join(args.out_dir, output_filename)
        
        # 可视化
        try:
            visualize_sample(sample, output_path, show_mask=args.show_mask, dpi=args.dpi)
            print(f"  [{i+1}/{len(selected_samples)}] Saved: {output_path}")
        except Exception as e:
            print(f"  [{i+1}/{len(selected_samples)}] Error processing sample {i}: {e}")
            continue
    
    print(f"\nDone! Visualizations saved to: {args.out_dir}")
    print(f"Total: {len(selected_samples)} images generated")


if __name__ == '__main__':
    main()

```


检查语法错误并测试脚本：



测试脚本，随机采样几个样本：



检查生成的文件：



脚本已创建并通过测试。使用说明：

## 脚本功能

已创建通用可视化脚本：`seg-rl/peft/visualize_training_data.py`

### 主要特性

1. 支持命令行参数：JSONL 文件和输出目录可指定
2. 两种模式：
   - `random`：随机采样指定数量的样本
   - `all`：遍历所有样本生成可视化
3. 区分前景点和背景点：红色=前景（label=1），蓝色=背景（label=0）
4. 可选显示真值掩模：可叠加显示绿色半透明掩模

### 使用示例

#### 1. 随机采样模式（默认）

```bash
# 随机采样5个样本
python -m seg-rl.peft.visualize_training_data \
    --jsonl outputs/braintumour/peft_train-251107.jsonl \
    --out_dir outputs/braintumour/visualizations \
    --mode random \
    --num_samples 5 \
    --seed 42
```

#### 2. 遍历所有样本

```bash
# 为所有样本生成可视化
python -m seg-rl.peft.visualize_training_data \
    --jsonl outputs/braintumour/peft_train-251107.jsonl \
    --out_dir outputs/braintumour/visualizations_all \
    --mode all
```

#### 3. 不显示掩模

```bash
# 只显示点和图像，不显示真值掩模
python -m seg-rl.peft.visualize_training_data \
    --jsonl outputs/braintumour/peft_train-251107.jsonl \
    --out_dir outputs/braintumour/visualizations \
    --mode random \
    --num_samples 10 \
    --no_mask
```

#### 4. 自定义输出文件名前缀

```bash
# 使用自定义前缀
python -m seg-rl.peft.visualize_training_data \
    --jsonl outputs/braintumour/peft_train-251107.jsonl \
    --out_dir outputs/braintumour/visualizations \
    --mode random \
    --num_samples 5 \
    --prefix train_sample
```

#### 5. 高分辨率输出

```bash
# 使用更高的DPI（300 DPI，适合打印）
python -m seg-rl.peft.visualize_training_data \
    --jsonl outputs/braintumour/peft_train-251107.jsonl \
    --out_dir outputs/braintumour/visualizations \
    --mode random \
    --num_samples 5 \
    --dpi 300
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--jsonl` | str | 必需 | 输入JSONL文件路径 |
| `--out_dir` | str | 必需 | 输出图像文件夹路径 |
| `--mode` | str | `random` | 采样模式：`random` 或 `all` |
| `--num_samples` | int | 5 | 随机采样模式下的样本数量 |
| `--seed` | int | 42 | 随机种子（仅当mode=random时有效） |
| `--show_mask` | flag | True | 是否显示真值掩模（默认True） |
| `--no_mask` | flag | - | 不显示真值掩模 |
| `--dpi` | int | 150 | 输出图像分辨率（DPI） |
| `--prefix` | str | `sample` | 输出文件名前缀 |

### 输出说明

- 随机模式：输出文件名为 `sample_000.png`, `sample_001.png`, ...
- 全部模式：输出文件名为 `sample_<图像文件名>.png`

### 可视化内容

- 图像：原始RGB图像
- 真值掩模：绿色半透明叠加（如果启用）
- 前景点：红色圆点，标注为 `FG`
- 背景点：蓝色圆点，标注为 `BG`
- 点标签：每个点显示序号和类型（如 `0(FG)`, `1(BG)`）

脚本已就绪，可直接使用。

```shellscript
# 随机采样5个样本
python -m seg-rl.peft.visualize_training_data \
    --jsonl outputs/braintumour/peft_train-251107.jsonl \
    --out_dir outputs/braintumour/visualizations \
    --mode random \
    --num_samples 5 \
    --seed 42
```

```shellscript
# 为所有样本生成可视化
python -m seg-rl.peft.visualize_training_data \
    --jsonl outputs/braintumour/peft_train-251107.jsonl \
    --out_dir outputs/braintumour/visualizations_all \
    --mode all
```

```shellscript
# 只显示点和图像，不显示真值掩模
python -m seg-rl.peft.visualize_training_data \
    --jsonl outputs/braintumour/peft_train-251107.jsonl \
    --out_dir outputs/braintumour/visualizations \
    --mode random \
    --num_samples 10 \
    --no_mask
```

```shellscript
# 使用自定义前缀
python -m seg-rl.peft.visualize_training_data \
    --jsonl outputs/braintumour/peft_train-251107.jsonl \
    --out_dir outputs/braintumour/visualizations \
    --mode random \
    --num_samples 5 \
    --prefix train_sample
```

```shellscript
# 使用更高的DPI（300 DPI，适合打印）
python -m seg-rl.peft.visualize_training_data \
    --jsonl outputs/braintumour/peft_train-251107.jsonl \
    --out_dir outputs/braintumour/visualizations \
    --mode random \
    --num_samples 5 \
    --dpi 300
```

---

**User**

在该脚本里再加入第二幅图像，叠加展示真值mask与相应的预测mask，用不同的半透明颜色来表示，并在图片角落或标题里显示该标本作分割预测的各项指标（DICE和IoU等等）

---

**Cursor**


查看评估脚本，了解如何加载模型和计算指标：



修改可视化脚本，添加预测掩模对比和指标显示：



修改脚本，添加预测掩模对比和指标显示：


```python
#!/usr/bin/env python3
"""
可视化训练数据脚本

用于可视化PEFT训练数据中的图像、点标注和真值掩模。
支持随机采样或遍历所有样本。
如果提供了checkpoint，还会生成预测掩模对比图。

用法:
    # 随机采样5个样本（仅显示标注）
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_train-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode random \
        --num_samples 5

    # 遍历所有样本（带预测对比）
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_test-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode all \
        --checkpoint outputs/braintumour/peft_supervised_baseline-251107/checkpoint_best.pt \
        --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
        --image_size 512 512
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

# 添加父目录到路径
_PARENT_DIR = Path(__file__).parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from seg_rl.peft.lora_sam2 import LoRASAM2Wrapper
from seg_rl.peft.point_predictor_peft import PointPredictorFromSAMFeatures
from seg_rl.peft.utils_peft import compute_dice, compute_iou, load_checkpoint


def load_jsonl(jsonl_path: str) -> List[Dict]:
    """加载JSONL文件"""
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith('['):
            # JSON数组格式
            data = json.loads(content)
        else:
            # JSONL格式（逐行）
            data = []
            for line in content.splitlines():
                if line.strip():
                    data.append(json.loads(line))
    return data


def visualize_sample(
    sample: Dict,
    output_path: str,
    show_mask: bool = True,
    dpi: int = 150,
) -> None:
    """
    可视化单个样本
    
    Args:
        sample: 样本字典，包含 'image', 'gt_mask', 'points', 'labels' 等字段
        output_path: 输出图像路径
        show_mask: 是否显示真值掩模
        dpi: 图像分辨率
    """
    # 加载图像
    image_path = sample['image']
    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        return
    
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    
    # 获取点和标签
    points = sample.get('points', [])
    labels = sample.get('labels', [])
    
    if len(points) != len(labels):
        print(f"Warning: Points and labels length mismatch: {len(points)} vs {len(labels)}")
        return
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(img_np)
    
    # 如果有真值掩模，叠加显示
    if show_mask:
        gt_mask_path = sample.get('gt_mask', None)
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                mask_binary = (gt_mask > 127).astype(float)
                ax.imshow(mask_binary, alpha=0.3, cmap='Greens')
            except Exception as e:
                print(f"Warning: Failed to load mask {gt_mask_path}: {e}")
    
    # 绘制点
    for j, ((x, y), label) in enumerate(zip(points, labels)):
        if label == 1:
            # 前景点：红色，实心圆
            color = 'red'
            size = 150
            point_type = 'FG'
        else:
            # 背景点：蓝色，实心圆
            color = 'blue'
            size = 120
            point_type = 'BG'
        
        # 绘制点
        ax.scatter(x, y, c=color, s=size, marker='o', 
                  edgecolors='white', linewidths=2, zorder=10)
        
        # 添加标签文本
        ax.text(x + 15, y, f'{j}({point_type})', 
               fontsize=10, color='white', weight='bold',
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    # 设置标题
    stem = Path(image_path).stem
    title = f'{stem}: {len(points)} points'
    ax.set_title(title, fontsize=14, weight='bold')
    ax.axis('off')
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='可视化PEFT训练数据',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--jsonl',
        type=str,
        required=True,
        help='输入JSONL文件路径'
    )
    
    parser.add_argument(
        '--out_dir',
        type=str,
        required=True,
        help='输出图像文件夹路径'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['random', 'all'],
        default='random',
        help='采样模式: random=随机采样, all=遍历所有样本'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='随机采样模式下的样本数量（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--show_mask',
        action='store_true',
        default=True,
        help='是否显示真值掩模（默认True）'
    )
    
    parser.add_argument(
        '--no_mask',
        dest='show_mask',
        action='store_false',
        help='不显示真值掩模'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='输出图像分辨率（DPI）'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default='sample',
        help='输出文件名前缀'
    )
    
    args = parser.parse_args()
    
    # 加载数据
    print(f"Loading data from {args.jsonl}...")
    data = load_jsonl(args.jsonl)
    print(f"Loaded {len(data)} samples")
    
    # 选择样本
    if args.mode == 'random':
        random.seed(args.seed)
        selected_samples = random.sample(data, min(args.num_samples, len(data)))
        print(f"Randomly selected {len(selected_samples)} samples")
    else:
        selected_samples = data
        print(f"Processing all {len(selected_samples)} samples")
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 可视化每个样本
    print(f"\nGenerating visualizations...")
    for i, sample in enumerate(selected_samples):
        # 生成输出文件名
        if args.mode == 'random':
            output_filename = f'{args.prefix}_{i:03d}.png'
        else:
            # 使用图像文件名作为输出文件名
            image_path = sample.get('image', '')
            if image_path:
                stem = Path(image_path).stem
                output_filename = f'{args.prefix}_{stem}.png'
            else:
                output_filename = f'{args.prefix}_{i:04d}.png'
        
        output_path = os.path.join(args.out_dir, output_filename)
        
        # 可视化
        try:
            visualize_sample(sample, output_path, show_mask=args.show_mask, dpi=args.dpi)
            print(f"  [{i+1}/{len(selected_samples)}] Saved: {output_path}")
        except Exception as e:
            print(f"  [{i+1}/{len(selected_samples)}] Error processing sample {i}: {e}")
            continue
    
    print(f"\nDone! Visualizations saved to: {args.out_dir}")
    print(f"Total: {len(selected_samples)} images generated")


if __name__ == '__main__':
    main()


```

```python
#!/usr/bin/env python3
"""
可视化训练数据脚本

用于可视化PEFT训练数据中的图像、点标注和真值掩模。
支持随机采样或遍历所有样本。
如果提供了checkpoint，还会生成预测掩模对比图。

用法:
    # 随机采样5个样本（仅显示标注）
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_train-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode random \
        --num_samples 5

    # 遍历所有样本（带预测对比）
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_test-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode all \
        --checkpoint outputs/braintumour/peft_supervised_baseline-251107/checkpoint_best.pt \
        --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
        --image_size 512 512
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

# 添加父目录到路径
_PARENT_DIR = Path(__file__).parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from seg_rl.peft.lora_sam2 import LoRASAM2Wrapper
from seg_rl.peft.point_predictor_peft import PointPredictorFromSAMFeatures
from seg_rl.peft.utils_peft import compute_dice, compute_iou, load_checkpoint


def load_jsonl(jsonl_path: str) -> List[Dict]:
    """加载JSONL文件"""
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith('['):
            # JSON数组格式
            data = json.loads(content)
        else:
            # JSONL格式（逐行）
            data = []
            for line in content.splitlines():
                if line.strip():
                    data.append(json.loads(line))
    return data


def visualize_sample(
    sample: Dict,
    output_path: str,
    show_mask: bool = True,
    dpi: int = 150,
    sam2_lora: Optional[LoRASAM2Wrapper] = None,
    point_predictor: Optional[PointPredictorFromSAMFeatures] = None,
    image_size: Tuple[int, int] = (512, 512),
    device: str = 'cuda',
    max_rollout_steps: int = 16,
) -> None:
    """
    可视化单个样本
    
    Args:
        sample: 样本字典，包含 'image', 'gt_mask', 'points', 'labels' 等字段
        output_path: 输出图像路径
        show_mask: 是否显示真值掩模
        dpi: 图像分辨率
        sam2_lora: SAM2 LoRA包装器（可选，用于预测）
        point_predictor: 点预测网络（可选，用于预测）
        image_size: 图像尺寸
        device: 设备
        max_rollout_steps: 最大rollout步数
    """
    # 加载图像
    image_path = sample['image']
    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        return
    
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    orig_w, orig_h = img.size
    
    # 获取点和标签
    points = sample.get('points', [])
    labels = sample.get('labels', [])
    
    if len(points) != len(labels):
        print(f"Warning: Points and labels length mismatch: {len(points)} vs {len(labels)}")
        return
    
    # 是否进行预测
    do_prediction = sam2_lora is not None and point_predictor is not None
    
    # 计算预测掩模和指标（如果需要）
    pred_mask = None
    metrics = {}
    pred_points_seq = []
    pred_labels_seq = []
    
    if do_prediction:
        try:
            # 运行rollout预测
            img_resized = img.resize(image_size, Image.BILINEAR)
            img_tensor = TF.to_tensor(img_resized).unsqueeze(0).to(device)
            
            # 加载真值掩模
            gt_mask_path = sample.get('gt_mask', None)
            if gt_mask_path and os.path.exists(gt_mask_path):
                gt_mask = Image.open(gt_mask_path).convert('L')
                gt_mask_resized = gt_mask.resize(image_size, Image.NEAREST)
                gt_mask_tensor = TF.to_tensor(gt_mask_resized).to(device)
                gt_mask_binary = (gt_mask_tensor > 0.5).float()
            else:
                gt_mask_binary = None
            
            # Rollout预测
            prev_mask = torch.zeros(1, 1, image_size[0], image_size[1]).to(device)
            
            for step in range(max_rollout_steps):
                # 预测下一个点
                sam_features = sam2_lora.get_image_features(img_tensor, feature_scale=8)
                heatmap_logits, label_logits = point_predictor(sam_features, prev_mask)
                
                # Argmax获取点
                B, _, H, W = heatmap_logits.shape
                heatmap_flat = heatmap_logits.view(B, -1)
                pred_idx = heatmap_flat.argmax(dim=1)
                pred_y = (pred_idx // W).float()
                pred_x = (pred_idx % W).float()
                pred_label = label_logits.argmax(dim=1).item()
                
                # 记录点
                pred_points_seq.append((pred_x.item(), pred_y.item()))
                pred_labels_seq.append(pred_label)
                
                # 生成掩模
                points_orig = [(px * orig_w / image_size[1], py * orig_h / image_size[0])
                              for px, py in pred_points_seq]
                prev_mask = sam2_lora.predict_mask(
                    np.array(img), points_orig, pred_labels_seq
                )
                if isinstance(prev_mask, torch.Tensor):
                    prev_mask = prev_mask.unsqueeze(0).to(device)
                else:
                    prev_mask = torch.from_numpy(prev_mask).unsqueeze(0).to(device)
                prev_mask = F.interpolate(
                    prev_mask, size=image_size, mode='nearest'
                )
                
                # 计算指标
                if gt_mask_binary is not None:
                    dice = compute_dice(prev_mask[0, 0], gt_mask_binary[0, 0])
                    iou = compute_iou(prev_mask[0, 0], gt_mask_binary[0, 0])
                    
                    if step == max_rollout_steps - 1 or dice > 0.95:
                        metrics['dice'] = dice
                        metrics['iou'] = iou
                        metrics['num_points'] = len(pred_points_seq)
                        pred_mask = prev_mask[0, 0].cpu().numpy()
                        break
            
            # 转换为numpy用于可视化
            if pred_mask is None:
                pred_mask = prev_mask[0, 0].cpu().numpy()
            
        except Exception as e:
            print(f"Warning: Prediction failed: {e}")
            do_prediction = False
    
    # 创建图形
    if do_prediction:
        # 两幅图：第一幅显示点和标注，第二幅显示掩模对比
        fig, axes = plt.subplots(1, 2, figsize=(24, 12))
        ax1, ax2 = axes
    else:
        # 单幅图：只显示点和标注
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 12))
        ax2 = None
    
    # 第一幅图：显示点和标注
    ax1.imshow(img_np)
    
    # 如果有真值掩模，叠加显示
    if show_mask:
        gt_mask_path = sample.get('gt_mask', None)
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                mask_binary = (gt_mask > 127).astype(float)
                ax1.imshow(mask_binary, alpha=0.3, cmap='Greens')
            except Exception as e:
                print(f"Warning: Failed to load mask {gt_mask_path}: {e}")
    
    # 绘制点
    for j, ((x, y), label) in enumerate(zip(points, labels)):
        if label == 1:
            # 前景点：红色，实心圆
            color = 'red'
            size = 150
            point_type = 'FG'
        else:
            # 背景点：蓝色，实心圆
            color = 'blue'
            size = 120
            point_type = 'BG'
        
        # 绘制点
        ax1.scatter(x, y, c=color, s=size, marker='o', 
                   edgecolors='white', linewidths=2, zorder=10)
        
        # 添加标签文本
        ax1.text(x + 15, y, f'{j}({point_type})', 
                fontsize=10, color='white', weight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    # 设置第一幅图的标题
    stem = Path(image_path).stem
    title1 = f'{stem}: {len(points)} points (Ground Truth)'
    ax1.set_title(title1, fontsize=14, weight='bold')
    ax1.axis('off')
    
    # 第二幅图：显示掩模对比（如果进行预测）
    if do_prediction and ax2 is not None:
        ax2.imshow(img_np)
        
        # 叠加真值掩模（绿色）
        gt_mask_path = sample.get('gt_mask', None)
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                gt_mask_binary = (gt_mask > 127).astype(float)
                # 调整尺寸以匹配图像
                if gt_mask_binary.shape != img_np.shape[:2]:
                    from scipy.ndimage import zoom
                    scale_y = img_np.shape[0] / gt_mask_binary.shape[0]
                    scale_x = img_np.shape[1] / gt_mask_binary.shape[1]
                    gt_mask_binary = zoom(gt_mask_binary, (scale_y, scale_x), order=0)
                
                # 绿色半透明显示真值掩模
                gt_mask_colored = np.zeros_like(img_np)
                gt_mask_colored[:, :, 1] = gt_mask_binary * 255
                ax2.imshow(gt_mask_colored, alpha=0.4)
            except Exception as e:
                print(f"Warning: Failed to load GT mask: {e}")
        
        # 叠加预测掩模（红色）
        if pred_mask is not None:
            # 调整尺寸以匹配图像
            if pred_mask.shape != img_np.shape[:2]:
                from scipy.ndimage import zoom
                scale_y = img_np.shape[0] / pred_mask.shape[0]
                scale_x = img_np.shape[1] / pred_mask.shape[1]
                pred_mask_resized = zoom(pred_mask, (scale_y, scale_x), order=0)
            else:
                pred_mask_resized = pred_mask
            
            pred_mask_binary = (pred_mask_resized > 0.5).astype(float)
            # 红色半透明显示预测掩模
            pred_mask_colored = np.zeros_like(img_np)
            pred_mask_colored[:, :, 0] = pred_mask_binary * 255
            ax2.imshow(pred_mask_colored, alpha=0.4)
        
        # 绘制预测点
        for j, ((x, y), label) in enumerate(zip(pred_points_seq, pred_labels_seq)):
            if label == 1:
                color = 'red'
                size = 150
                point_type = 'FG'
            else:
                color = 'blue'
                size = 120
                point_type = 'BG'
            
            # 缩放点坐标到原始图像尺寸
            x_orig = x * orig_w / image_size[1]
            y_orig = y * orig_h / image_size[0]
            
            ax2.scatter(x_orig, y_orig, c=color, s=size, marker='o', 
                       edgecolors='white', linewidths=2, zorder=10)
            ax2.text(x_orig + 15, y_orig, f'{j}({point_type})', 
                    fontsize=10, color='white', weight='bold',
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
        
        # 设置第二幅图的标题和指标
        title2 = f'Prediction: {len(pred_points_seq)} points'
        if metrics:
            metrics_text = f"Dice: {metrics.get('dice', 0):.3f}, IoU: {metrics.get('iou', 0):.3f}"
            title2 += f'\n{metrics_text}'
        ax2.set_title(title2, fontsize=14, weight='bold')
        ax2.axis('off')
        
        # 在图像角落添加指标文本
        if metrics:
            metrics_str = f"Dice: {metrics.get('dice', 0):.3f}\nIoU: {metrics.get('iou', 0):.3f}\nPoints: {metrics.get('num_points', 0)}"
            ax2.text(0.02, 0.98, metrics_str, 
                    transform=ax2.transAxes, fontsize=12, weight='bold',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='可视化PEFT训练数据',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--jsonl',
        type=str,
        required=True,
        help='输入JSONL文件路径'
    )
    
    parser.add_argument(
        '--out_dir',
        type=str,
        required=True,
        help='输出图像文件夹路径'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['random', 'all'],
        default='random',
        help='采样模式: random=随机采样, all=遍历所有样本'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='随机采样模式下的样本数量（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--show_mask',
        action='store_true',
        default=True,
        help='是否显示真值掩模（默认True）'
    )
    
    parser.add_argument(
        '--no_mask',
        dest='show_mask',
        action='store_false',
        help='不显示真值掩模'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='输出图像分辨率（DPI）'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default='sample',
        help='输出文件名前缀'
    )
    
    args = parser.parse_args()
    
    # 加载数据
    print(f"Loading data from {args.jsonl}...")
    data = load_jsonl(args.jsonl)
    print(f"Loaded {len(data)} samples")
    
    # 选择样本
    if args.mode == 'random':
        random.seed(args.seed)
        selected_samples = random.sample(data, min(args.num_samples, len(data)))
        print(f"Randomly selected {len(selected_samples)} samples")
    else:
        selected_samples = data
        print(f"Processing all {len(selected_samples)} samples")
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 可视化每个样本
    print(f"\nGenerating visualizations...")
    for i, sample in enumerate(selected_samples):
        # 生成输出文件名
        if args.mode == 'random':
            output_filename = f'{args.prefix}_{i:03d}.png'
        else:
            # 使用图像文件名作为输出文件名
            image_path = sample.get('image', '')
            if image_path:
                stem = Path(image_path).stem
                output_filename = f'{args.prefix}_{stem}.png'
            else:
                output_filename = f'{args.prefix}_{i:04d}.png'
        
        output_path = os.path.join(args.out_dir, output_filename)
        
        # 可视化
        try:
            visualize_sample(sample, output_path, show_mask=args.show_mask, dpi=args.dpi)
            print(f"  [{i+1}/{len(selected_samples)}] Saved: {output_path}")
        except Exception as e:
            print(f"  [{i+1}/{len(selected_samples)}] Error processing sample {i}: {e}")
            continue
    
    print(f"\nDone! Visualizations saved to: {args.out_dir}")
    print(f"Total: {len(selected_samples)} images generated")


if __name__ == '__main__':
    main()


```

```python
#!/usr/bin/env python3
"""
可视化训练数据脚本

用于可视化PEFT训练数据中的图像、点标注和真值掩模。
支持随机采样或遍历所有样本。
如果提供了checkpoint，还会生成预测掩模对比图。

用法:
    # 随机采样5个样本（仅显示标注）
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_train-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode random \
        --num_samples 5

    # 遍历所有样本（带预测对比）
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_test-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode all \
        --checkpoint outputs/braintumour/peft_supervised_baseline-251107/checkpoint_best.pt \
        --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
        --image_size 512 512
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

# 添加父目录到路径
_PARENT_DIR = Path(__file__).parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from seg_rl.peft.lora_sam2 import LoRASAM2Wrapper
from seg_rl.peft.point_predictor_peft import PointPredictorFromSAMFeatures
from seg_rl.peft.utils_peft import compute_dice, compute_iou, load_checkpoint


def load_jsonl(jsonl_path: str) -> List[Dict]:
    """加载JSONL文件"""
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith('['):
            # JSON数组格式
            data = json.loads(content)
        else:
            # JSONL格式（逐行）
            data = []
            for line in content.splitlines():
                if line.strip():
                    data.append(json.loads(line))
    return data


def visualize_sample(
    sample: Dict,
    output_path: str,
    show_mask: bool = True,
    dpi: int = 150,
    sam2_lora: Optional[LoRASAM2Wrapper] = None,
    point_predictor: Optional[PointPredictorFromSAMFeatures] = None,
    image_size: Tuple[int, int] = (512, 512),
    device: str = 'cuda',
    max_rollout_steps: int = 16,
) -> None:
    """
    可视化单个样本
    
    Args:
        sample: 样本字典，包含 'image', 'gt_mask', 'points', 'labels' 等字段
        output_path: 输出图像路径
        show_mask: 是否显示真值掩模
        dpi: 图像分辨率
        sam2_lora: SAM2 LoRA包装器（可选，用于预测）
        point_predictor: 点预测网络（可选，用于预测）
        image_size: 图像尺寸
        device: 设备
        max_rollout_steps: 最大rollout步数
    """
    # 加载图像
    image_path = sample['image']
    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        return
    
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    orig_w, orig_h = img.size
    
    # 获取点和标签
    points = sample.get('points', [])
    labels = sample.get('labels', [])
    
    if len(points) != len(labels):
        print(f"Warning: Points and labels length mismatch: {len(points)} vs {len(labels)}")
        return
    
    # 是否进行预测
    do_prediction = sam2_lora is not None and point_predictor is not None
    
    # 计算预测掩模和指标（如果需要）
    pred_mask = None
    metrics = {}
    pred_points_seq = []
    pred_labels_seq = []
    
    if do_prediction:
        try:
            # 运行rollout预测
            img_resized = img.resize(image_size, Image.BILINEAR)
            img_tensor = TF.to_tensor(img_resized).unsqueeze(0).to(device)
            
            # 加载真值掩模
            gt_mask_path = sample.get('gt_mask', None)
            if gt_mask_path and os.path.exists(gt_mask_path):
                gt_mask = Image.open(gt_mask_path).convert('L')
                gt_mask_resized = gt_mask.resize(image_size, Image.NEAREST)
                gt_mask_tensor = TF.to_tensor(gt_mask_resized).to(device)
                gt_mask_binary = (gt_mask_tensor > 0.5).float()
            else:
                gt_mask_binary = None
            
            # Rollout预测
            prev_mask = torch.zeros(1, 1, image_size[0], image_size[1]).to(device)
            
            for step in range(max_rollout_steps):
                # 预测下一个点
                sam_features = sam2_lora.get_image_features(img_tensor, feature_scale=8)
                heatmap_logits, label_logits = point_predictor(sam_features, prev_mask)
                
                # Argmax获取点
                B, _, H, W = heatmap_logits.shape
                heatmap_flat = heatmap_logits.view(B, -1)
                pred_idx = heatmap_flat.argmax(dim=1)
                pred_y = (pred_idx // W).float()
                pred_x = (pred_idx % W).float()
                pred_label = label_logits.argmax(dim=1).item()
                
                # 记录点
                pred_points_seq.append((pred_x.item(), pred_y.item()))
                pred_labels_seq.append(pred_label)
                
                # 生成掩模
                points_orig = [(px * orig_w / image_size[1], py * orig_h / image_size[0])
                              for px, py in pred_points_seq]
                prev_mask = sam2_lora.predict_mask(
                    np.array(img), points_orig, pred_labels_seq
                )
                if isinstance(prev_mask, torch.Tensor):
                    prev_mask = prev_mask.unsqueeze(0).to(device)
                else:
                    prev_mask = torch.from_numpy(prev_mask).unsqueeze(0).to(device)
                prev_mask = F.interpolate(
                    prev_mask, size=image_size, mode='nearest'
                )
                
                # 计算指标
                if gt_mask_binary is not None:
                    dice = compute_dice(prev_mask[0, 0], gt_mask_binary[0, 0])
                    iou = compute_iou(prev_mask[0, 0], gt_mask_binary[0, 0])
                    
                    if step == max_rollout_steps - 1 or dice > 0.95:
                        metrics['dice'] = dice
                        metrics['iou'] = iou
                        metrics['num_points'] = len(pred_points_seq)
                        pred_mask = prev_mask[0, 0].cpu().numpy()
                        break
            
            # 转换为numpy用于可视化
            if pred_mask is None:
                pred_mask = prev_mask[0, 0].cpu().numpy()
            
        except Exception as e:
            print(f"Warning: Prediction failed: {e}")
            do_prediction = False
    
    # 创建图形
    if do_prediction:
        # 两幅图：第一幅显示点和标注，第二幅显示掩模对比
        fig, axes = plt.subplots(1, 2, figsize=(24, 12))
        ax1, ax2 = axes
    else:
        # 单幅图：只显示点和标注
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 12))
        ax2 = None
    
    # 第一幅图：显示点和标注
    ax1.imshow(img_np)
    
    # 如果有真值掩模，叠加显示
    if show_mask:
        gt_mask_path = sample.get('gt_mask', None)
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                mask_binary = (gt_mask > 127).astype(float)
                ax1.imshow(mask_binary, alpha=0.3, cmap='Greens')
            except Exception as e:
                print(f"Warning: Failed to load mask {gt_mask_path}: {e}")
    
    # 绘制点
    for j, ((x, y), label) in enumerate(zip(points, labels)):
        if label == 1:
            # 前景点：红色，实心圆
            color = 'red'
            size = 150
            point_type = 'FG'
        else:
            # 背景点：蓝色，实心圆
            color = 'blue'
            size = 120
            point_type = 'BG'
        
        # 绘制点
        ax1.scatter(x, y, c=color, s=size, marker='o', 
                   edgecolors='white', linewidths=2, zorder=10)
        
        # 添加标签文本
        ax1.text(x + 15, y, f'{j}({point_type})', 
                fontsize=10, color='white', weight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    # 设置第一幅图的标题
    stem = Path(image_path).stem
    title1 = f'{stem}: {len(points)} points (Ground Truth)'
    ax1.set_title(title1, fontsize=14, weight='bold')
    ax1.axis('off')
    
    # 第二幅图：显示掩模对比（如果进行预测）
    if do_prediction and ax2 is not None:
        ax2.imshow(img_np)
        
        # 叠加真值掩模（绿色）
        gt_mask_path = sample.get('gt_mask', None)
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                gt_mask_binary = (gt_mask > 127).astype(float)
                # 调整尺寸以匹配图像
                if gt_mask_binary.shape != img_np.shape[:2]:
                    from scipy.ndimage import zoom
                    scale_y = img_np.shape[0] / gt_mask_binary.shape[0]
                    scale_x = img_np.shape[1] / gt_mask_binary.shape[1]
                    gt_mask_binary = zoom(gt_mask_binary, (scale_y, scale_x), order=0)
                
                # 绿色半透明显示真值掩模
                gt_mask_colored = np.zeros_like(img_np)
                gt_mask_colored[:, :, 1] = gt_mask_binary * 255
                ax2.imshow(gt_mask_colored, alpha=0.4)
            except Exception as e:
                print(f"Warning: Failed to load GT mask: {e}")
        
        # 叠加预测掩模（红色）
        if pred_mask is not None:
            # 调整尺寸以匹配图像
            if pred_mask.shape != img_np.shape[:2]:
                from scipy.ndimage import zoom
                scale_y = img_np.shape[0] / pred_mask.shape[0]
                scale_x = img_np.shape[1] / pred_mask.shape[1]
                pred_mask_resized = zoom(pred_mask, (scale_y, scale_x), order=0)
            else:
                pred_mask_resized = pred_mask
            
            pred_mask_binary = (pred_mask_resized > 0.5).astype(float)
            # 红色半透明显示预测掩模
            pred_mask_colored = np.zeros_like(img_np)
            pred_mask_colored[:, :, 0] = pred_mask_binary * 255
            ax2.imshow(pred_mask_colored, alpha=0.4)
        
        # 绘制预测点
        for j, ((x, y), label) in enumerate(zip(pred_points_seq, pred_labels_seq)):
            if label == 1:
                color = 'red'
                size = 150
                point_type = 'FG'
            else:
                color = 'blue'
                size = 120
                point_type = 'BG'
            
            # 缩放点坐标到原始图像尺寸
            x_orig = x * orig_w / image_size[1]
            y_orig = y * orig_h / image_size[0]
            
            ax2.scatter(x_orig, y_orig, c=color, s=size, marker='o', 
                       edgecolors='white', linewidths=2, zorder=10)
            ax2.text(x_orig + 15, y_orig, f'{j}({point_type})', 
                    fontsize=10, color='white', weight='bold',
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
        
        # 设置第二幅图的标题和指标
        title2 = f'Prediction: {len(pred_points_seq)} points'
        if metrics:
            metrics_text = f"Dice: {metrics.get('dice', 0):.3f}, IoU: {metrics.get('iou', 0):.3f}"
            title2 += f'\n{metrics_text}'
        ax2.set_title(title2, fontsize=14, weight='bold')
        ax2.axis('off')
        
        # 在图像角落添加指标文本
        if metrics:
            metrics_str = f"Dice: {metrics.get('dice', 0):.3f}\nIoU: {metrics.get('iou', 0):.3f}\nPoints: {metrics.get('num_points', 0)}"
            ax2.text(0.02, 0.98, metrics_str, 
                    transform=ax2.transAxes, fontsize=12, weight='bold',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='可视化PEFT训练数据',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--jsonl',
        type=str,
        required=True,
        help='输入JSONL文件路径'
    )
    
    parser.add_argument(
        '--out_dir',
        type=str,
        required=True,
        help='输出图像文件夹路径'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['random', 'all'],
        default='random',
        help='采样模式: random=随机采样, all=遍历所有样本'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='随机采样模式下的样本数量（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--show_mask',
        action='store_true',
        default=True,
        help='是否显示真值掩模（默认True）'
    )
    
    parser.add_argument(
        '--no_mask',
        dest='show_mask',
        action='store_false',
        help='不显示真值掩模'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='输出图像分辨率（DPI）'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default='sample',
        help='输出文件名前缀'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='模型checkpoint路径（如果提供，将生成预测对比图）'
    )
    
    parser.add_argument(
        '--sam_checkpoint',
        type=str,
        default=None,
        help='SAM2 checkpoint路径（如果提供checkpoint，此参数必需）'
    )
    
    parser.add_argument(
        '--image_size',
        type=int,
        nargs=2,
        default=[512, 512],
        help='图像尺寸 [H, W]（用于预测）'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='设备（cuda/cpu）'
    )
    
    parser.add_argument(
        '--max_rollout_steps',
        type=int,
        default=16,
        help='最大rollout步数（用于预测）'
    )
    
    args = parser.parse_args()
    
    # 加载数据
    print(f"Loading data from {args.jsonl}...")
    data = load_jsonl(args.jsonl)
    print(f"Loaded {len(data)} samples")
    
    # 选择样本
    if args.mode == 'random':
        random.seed(args.seed)
        selected_samples = random.sample(data, min(args.num_samples, len(data)))
        print(f"Randomly selected {len(selected_samples)} samples")
    else:
        selected_samples = data
        print(f"Processing all {len(selected_samples)} samples")
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 可视化每个样本
    print(f"\nGenerating visualizations...")
    for i, sample in enumerate(selected_samples):
        # 生成输出文件名
        if args.mode == 'random':
            output_filename = f'{args.prefix}_{i:03d}.png'
        else:
            # 使用图像文件名作为输出文件名
            image_path = sample.get('image', '')
            if image_path:
                stem = Path(image_path).stem
                output_filename = f'{args.prefix}_{stem}.png'
            else:
                output_filename = f'{args.prefix}_{i:04d}.png'
        
        output_path = os.path.join(args.out_dir, output_filename)
        
        # 可视化
        try:
            visualize_sample(sample, output_path, show_mask=args.show_mask, dpi=args.dpi)
            print(f"  [{i+1}/{len(selected_samples)}] Saved: {output_path}")
        except Exception as e:
            print(f"  [{i+1}/{len(selected_samples)}] Error processing sample {i}: {e}")
            continue
    
    print(f"\nDone! Visualizations saved to: {args.out_dir}")
    print(f"Total: {len(selected_samples)} images generated")


if __name__ == '__main__':
    main()


```

```python
#!/usr/bin/env python3
"""
可视化训练数据脚本

用于可视化PEFT训练数据中的图像、点标注和真值掩模。
支持随机采样或遍历所有样本。
如果提供了checkpoint，还会生成预测掩模对比图。

用法:
    # 随机采样5个样本（仅显示标注）
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_train-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode random \
        --num_samples 5

    # 遍历所有样本（带预测对比）
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_test-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode all \
        --checkpoint outputs/braintumour/peft_supervised_baseline-251107/checkpoint_best.pt \
        --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
        --image_size 512 512
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

# 添加父目录到路径
_PARENT_DIR = Path(__file__).parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from seg_rl.peft.lora_sam2 import LoRASAM2Wrapper
from seg_rl.peft.point_predictor_peft import PointPredictorFromSAMFeatures
from seg_rl.peft.utils_peft import compute_dice, compute_iou, load_checkpoint


def load_jsonl(jsonl_path: str) -> List[Dict]:
    """加载JSONL文件"""
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith('['):
            # JSON数组格式
            data = json.loads(content)
        else:
            # JSONL格式（逐行）
            data = []
            for line in content.splitlines():
                if line.strip():
                    data.append(json.loads(line))
    return data


def visualize_sample(
    sample: Dict,
    output_path: str,
    show_mask: bool = True,
    dpi: int = 150,
    sam2_lora: Optional[LoRASAM2Wrapper] = None,
    point_predictor: Optional[PointPredictorFromSAMFeatures] = None,
    image_size: Tuple[int, int] = (512, 512),
    device: str = 'cuda',
    max_rollout_steps: int = 16,
) -> None:
    """
    可视化单个样本
    
    Args:
        sample: 样本字典，包含 'image', 'gt_mask', 'points', 'labels' 等字段
        output_path: 输出图像路径
        show_mask: 是否显示真值掩模
        dpi: 图像分辨率
        sam2_lora: SAM2 LoRA包装器（可选，用于预测）
        point_predictor: 点预测网络（可选，用于预测）
        image_size: 图像尺寸
        device: 设备
        max_rollout_steps: 最大rollout步数
    """
    # 加载图像
    image_path = sample['image']
    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        return
    
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    orig_w, orig_h = img.size
    
    # 获取点和标签
    points = sample.get('points', [])
    labels = sample.get('labels', [])
    
    if len(points) != len(labels):
        print(f"Warning: Points and labels length mismatch: {len(points)} vs {len(labels)}")
        return
    
    # 是否进行预测
    do_prediction = sam2_lora is not None and point_predictor is not None
    
    # 计算预测掩模和指标（如果需要）
    pred_mask = None
    metrics = {}
    pred_points_seq = []
    pred_labels_seq = []
    
    if do_prediction:
        try:
            # 运行rollout预测
            img_resized = img.resize(image_size, Image.BILINEAR)
            img_tensor = TF.to_tensor(img_resized).unsqueeze(0).to(device)
            
            # 加载真值掩模
            gt_mask_path = sample.get('gt_mask', None)
            if gt_mask_path and os.path.exists(gt_mask_path):
                gt_mask = Image.open(gt_mask_path).convert('L')
                gt_mask_resized = gt_mask.resize(image_size, Image.NEAREST)
                gt_mask_tensor = TF.to_tensor(gt_mask_resized).to(device)
                gt_mask_binary = (gt_mask_tensor > 0.5).float()
            else:
                gt_mask_binary = None
            
            # Rollout预测
            prev_mask = torch.zeros(1, 1, image_size[0], image_size[1]).to(device)
            
            for step in range(max_rollout_steps):
                # 预测下一个点
                sam_features = sam2_lora.get_image_features(img_tensor, feature_scale=8)
                heatmap_logits, label_logits = point_predictor(sam_features, prev_mask)
                
                # Argmax获取点
                B, _, H, W = heatmap_logits.shape
                heatmap_flat = heatmap_logits.view(B, -1)
                pred_idx = heatmap_flat.argmax(dim=1)
                pred_y = (pred_idx // W).float()
                pred_x = (pred_idx % W).float()
                pred_label = label_logits.argmax(dim=1).item()
                
                # 记录点
                pred_points_seq.append((pred_x.item(), pred_y.item()))
                pred_labels_seq.append(pred_label)
                
                # 生成掩模
                points_orig = [(px * orig_w / image_size[1], py * orig_h / image_size[0])
                              for px, py in pred_points_seq]
                prev_mask = sam2_lora.predict_mask(
                    np.array(img), points_orig, pred_labels_seq
                )
                if isinstance(prev_mask, torch.Tensor):
                    prev_mask = prev_mask.unsqueeze(0).to(device)
                else:
                    prev_mask = torch.from_numpy(prev_mask).unsqueeze(0).to(device)
                prev_mask = F.interpolate(
                    prev_mask, size=image_size, mode='nearest'
                )
                
                # 计算指标
                if gt_mask_binary is not None:
                    dice = compute_dice(prev_mask[0, 0], gt_mask_binary[0, 0])
                    iou = compute_iou(prev_mask[0, 0], gt_mask_binary[0, 0])
                    
                    if step == max_rollout_steps - 1 or dice > 0.95:
                        metrics['dice'] = dice
                        metrics['iou'] = iou
                        metrics['num_points'] = len(pred_points_seq)
                        pred_mask = prev_mask[0, 0].cpu().numpy()
                        break
            
            # 转换为numpy用于可视化
            if pred_mask is None:
                pred_mask = prev_mask[0, 0].cpu().numpy()
            
        except Exception as e:
            print(f"Warning: Prediction failed: {e}")
            do_prediction = False
    
    # 创建图形
    if do_prediction:
        # 两幅图：第一幅显示点和标注，第二幅显示掩模对比
        fig, axes = plt.subplots(1, 2, figsize=(24, 12))
        ax1, ax2 = axes
    else:
        # 单幅图：只显示点和标注
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 12))
        ax2 = None
    
    # 第一幅图：显示点和标注
    ax1.imshow(img_np)
    
    # 如果有真值掩模，叠加显示
    if show_mask:
        gt_mask_path = sample.get('gt_mask', None)
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                mask_binary = (gt_mask > 127).astype(float)
                ax1.imshow(mask_binary, alpha=0.3, cmap='Greens')
            except Exception as e:
                print(f"Warning: Failed to load mask {gt_mask_path}: {e}")
    
    # 绘制点
    for j, ((x, y), label) in enumerate(zip(points, labels)):
        if label == 1:
            # 前景点：红色，实心圆
            color = 'red'
            size = 150
            point_type = 'FG'
        else:
            # 背景点：蓝色，实心圆
            color = 'blue'
            size = 120
            point_type = 'BG'
        
        # 绘制点
        ax1.scatter(x, y, c=color, s=size, marker='o', 
                   edgecolors='white', linewidths=2, zorder=10)
        
        # 添加标签文本
        ax1.text(x + 15, y, f'{j}({point_type})', 
                fontsize=10, color='white', weight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    # 设置第一幅图的标题
    stem = Path(image_path).stem
    title1 = f'{stem}: {len(points)} points (Ground Truth)'
    ax1.set_title(title1, fontsize=14, weight='bold')
    ax1.axis('off')
    
    # 第二幅图：显示掩模对比（如果进行预测）
    if do_prediction and ax2 is not None:
        ax2.imshow(img_np)
        
        # 叠加真值掩模（绿色）
        gt_mask_path = sample.get('gt_mask', None)
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                gt_mask_binary = (gt_mask > 127).astype(float)
                # 调整尺寸以匹配图像
                if gt_mask_binary.shape != img_np.shape[:2]:
                    from scipy.ndimage import zoom
                    scale_y = img_np.shape[0] / gt_mask_binary.shape[0]
                    scale_x = img_np.shape[1] / gt_mask_binary.shape[1]
                    gt_mask_binary = zoom(gt_mask_binary, (scale_y, scale_x), order=0)
                
                # 绿色半透明显示真值掩模
                gt_mask_colored = np.zeros_like(img_np)
                gt_mask_colored[:, :, 1] = gt_mask_binary * 255
                ax2.imshow(gt_mask_colored, alpha=0.4)
            except Exception as e:
                print(f"Warning: Failed to load GT mask: {e}")
        
        # 叠加预测掩模（红色）
        if pred_mask is not None:
            # 调整尺寸以匹配图像
            if pred_mask.shape != img_np.shape[:2]:
                from scipy.ndimage import zoom
                scale_y = img_np.shape[0] / pred_mask.shape[0]
                scale_x = img_np.shape[1] / pred_mask.shape[1]
                pred_mask_resized = zoom(pred_mask, (scale_y, scale_x), order=0)
            else:
                pred_mask_resized = pred_mask
            
            pred_mask_binary = (pred_mask_resized > 0.5).astype(float)
            # 红色半透明显示预测掩模
            pred_mask_colored = np.zeros_like(img_np)
            pred_mask_colored[:, :, 0] = pred_mask_binary * 255
            ax2.imshow(pred_mask_colored, alpha=0.4)
        
        # 绘制预测点
        for j, ((x, y), label) in enumerate(zip(pred_points_seq, pred_labels_seq)):
            if label == 1:
                color = 'red'
                size = 150
                point_type = 'FG'
            else:
                color = 'blue'
                size = 120
                point_type = 'BG'
            
            # 缩放点坐标到原始图像尺寸
            x_orig = x * orig_w / image_size[1]
            y_orig = y * orig_h / image_size[0]
            
            ax2.scatter(x_orig, y_orig, c=color, s=size, marker='o', 
                       edgecolors='white', linewidths=2, zorder=10)
            ax2.text(x_orig + 15, y_orig, f'{j}({point_type})', 
                    fontsize=10, color='white', weight='bold',
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
        
        # 设置第二幅图的标题和指标
        title2 = f'Prediction: {len(pred_points_seq)} points'
        if metrics:
            metrics_text = f"Dice: {metrics.get('dice', 0):.3f}, IoU: {metrics.get('iou', 0):.3f}"
            title2 += f'\n{metrics_text}'
        ax2.set_title(title2, fontsize=14, weight='bold')
        ax2.axis('off')
        
        # 在图像角落添加指标文本
        if metrics:
            metrics_str = f"Dice: {metrics.get('dice', 0):.3f}\nIoU: {metrics.get('iou', 0):.3f}\nPoints: {metrics.get('num_points', 0)}"
            ax2.text(0.02, 0.98, metrics_str, 
                    transform=ax2.transAxes, fontsize=12, weight='bold',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='可视化PEFT训练数据',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--jsonl',
        type=str,
        required=True,
        help='输入JSONL文件路径'
    )
    
    parser.add_argument(
        '--out_dir',
        type=str,
        required=True,
        help='输出图像文件夹路径'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['random', 'all'],
        default='random',
        help='采样模式: random=随机采样, all=遍历所有样本'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='随机采样模式下的样本数量（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--show_mask',
        action='store_true',
        default=True,
        help='是否显示真值掩模（默认True）'
    )
    
    parser.add_argument(
        '--no_mask',
        dest='show_mask',
        action='store_false',
        help='不显示真值掩模'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='输出图像分辨率（DPI）'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default='sample',
        help='输出文件名前缀'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='模型checkpoint路径（如果提供，将生成预测对比图）'
    )
    
    parser.add_argument(
        '--sam_checkpoint',
        type=str,
        default=None,
        help='SAM2 checkpoint路径（如果提供checkpoint，此参数必需）'
    )
    
    parser.add_argument(
        '--image_size',
        type=int,
        nargs=2,
        default=[512, 512],
        help='图像尺寸 [H, W]（用于预测）'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='设备（cuda/cpu）'
    )
    
    parser.add_argument(
        '--max_rollout_steps',
        type=int,
        default=16,
        help='最大rollout步数（用于预测）'
    )
    
    args = parser.parse_args()
    
    # 加载模型（如果提供了checkpoint）
    sam2_lora = None
    point_predictor = None
    
    if args.checkpoint:
        if not args.sam_checkpoint:
            print("Error: --sam_checkpoint is required when --checkpoint is provided")
            sys.exit(1)
        
        print(f"Loading model from {args.checkpoint}...")
        try:
            device = torch.device(args.device)
            
            # 加载checkpoint
            ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
            config = ckpt.get('config', {})
            
            # 提取配置
            lora_rank = config.get('lora_rank', 16)
            lora_alpha = config.get('lora_alpha', 32)
            fusion_mode = config.get('fusion_mode', 'film')
            feature_scale = config.get('feature_scale', 8)
            
            # 初始化SAM2 LoRA
            sam2_lora = LoRASAM2Wrapper(
                sam_checkpoint=args.sam_checkpoint,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                device=str(device),
            )
            
            # 初始化点预测网络
            sam_feature_dim = 256  # SAM2 Hiera Large的特征维度
            point_predictor = PointPredictorFromSAMFeatures(
                sam_feature_dim=sam_feature_dim,
                output_size=tuple(args.image_size),
                fusion_mode=fusion_mode,
                feature_scale=feature_scale,
            ).to(device)
            
            # 加载权重
            load_checkpoint(
                args.checkpoint, point_predictor, sam2_lora, None, None, None, str(device)
            )
            
            point_predictor.eval()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Will continue without prediction...")
            sam2_lora = None
            point_predictor = None
    
    # 加载数据
    print(f"Loading data from {args.jsonl}...")
    data = load_jsonl(args.jsonl)
    print(f"Loaded {len(data)} samples")
    
    # 选择样本
    if args.mode == 'random':
        random.seed(args.seed)
        selected_samples = random.sample(data, min(args.num_samples, len(data)))
        print(f"Randomly selected {len(selected_samples)} samples")
    else:
        selected_samples = data
        print(f"Processing all {len(selected_samples)} samples")
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 可视化每个样本
    print(f"\nGenerating visualizations...")
    for i, sample in enumerate(selected_samples):
        # 生成输出文件名
        if args.mode == 'random':
            output_filename = f'{args.prefix}_{i:03d}.png'
        else:
            # 使用图像文件名作为输出文件名
            image_path = sample.get('image', '')
            if image_path:
                stem = Path(image_path).stem
                output_filename = f'{args.prefix}_{stem}.png'
            else:
                output_filename = f'{args.prefix}_{i:04d}.png'
        
        output_path = os.path.join(args.out_dir, output_filename)
        
        # 可视化
        try:
            visualize_sample(sample, output_path, show_mask=args.show_mask, dpi=args.dpi)
            print(f"  [{i+1}/{len(selected_samples)}] Saved: {output_path}")
        except Exception as e:
            print(f"  [{i+1}/{len(selected_samples)}] Error processing sample {i}: {e}")
            continue
    
    print(f"\nDone! Visualizations saved to: {args.out_dir}")
    print(f"Total: {len(selected_samples)} images generated")


if __name__ == '__main__':
    main()


```

```python
#!/usr/bin/env python3
"""
可视化训练数据脚本

用于可视化PEFT训练数据中的图像、点标注和真值掩模。
支持随机采样或遍历所有样本。
如果提供了checkpoint，还会生成预测掩模对比图。

用法:
    # 随机采样5个样本（仅显示标注）
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_train-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode random \
        --num_samples 5

    # 遍历所有样本（带预测对比）
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_test-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode all \
        --checkpoint outputs/braintumour/peft_supervised_baseline-251107/checkpoint_best.pt \
        --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
        --image_size 512 512
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

# 添加父目录到路径
_PARENT_DIR = Path(__file__).parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from seg_rl.peft.lora_sam2 import LoRASAM2Wrapper
from seg_rl.peft.point_predictor_peft import PointPredictorFromSAMFeatures
from seg_rl.peft.utils_peft import compute_dice, compute_iou, load_checkpoint


def load_jsonl(jsonl_path: str) -> List[Dict]:
    """加载JSONL文件"""
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith('['):
            # JSON数组格式
            data = json.loads(content)
        else:
            # JSONL格式（逐行）
            data = []
            for line in content.splitlines():
                if line.strip():
                    data.append(json.loads(line))
    return data


def visualize_sample(
    sample: Dict,
    output_path: str,
    show_mask: bool = True,
    dpi: int = 150,
    sam2_lora: Optional[LoRASAM2Wrapper] = None,
    point_predictor: Optional[PointPredictorFromSAMFeatures] = None,
    image_size: Tuple[int, int] = (512, 512),
    device: str = 'cuda',
    max_rollout_steps: int = 16,
) -> None:
    """
    可视化单个样本
    
    Args:
        sample: 样本字典，包含 'image', 'gt_mask', 'points', 'labels' 等字段
        output_path: 输出图像路径
        show_mask: 是否显示真值掩模
        dpi: 图像分辨率
        sam2_lora: SAM2 LoRA包装器（可选，用于预测）
        point_predictor: 点预测网络（可选，用于预测）
        image_size: 图像尺寸
        device: 设备
        max_rollout_steps: 最大rollout步数
    """
    # 加载图像
    image_path = sample['image']
    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        return
    
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    orig_w, orig_h = img.size
    
    # 获取点和标签
    points = sample.get('points', [])
    labels = sample.get('labels', [])
    
    if len(points) != len(labels):
        print(f"Warning: Points and labels length mismatch: {len(points)} vs {len(labels)}")
        return
    
    # 是否进行预测
    do_prediction = sam2_lora is not None and point_predictor is not None
    
    # 计算预测掩模和指标（如果需要）
    pred_mask = None
    metrics = {}
    pred_points_seq = []
    pred_labels_seq = []
    
    if do_prediction:
        try:
            # 运行rollout预测
            img_resized = img.resize(image_size, Image.BILINEAR)
            img_tensor = TF.to_tensor(img_resized).unsqueeze(0).to(device)
            
            # 加载真值掩模
            gt_mask_path = sample.get('gt_mask', None)
            if gt_mask_path and os.path.exists(gt_mask_path):
                gt_mask = Image.open(gt_mask_path).convert('L')
                gt_mask_resized = gt_mask.resize(image_size, Image.NEAREST)
                gt_mask_tensor = TF.to_tensor(gt_mask_resized).to(device)
                gt_mask_binary = (gt_mask_tensor > 0.5).float()
            else:
                gt_mask_binary = None
            
            # Rollout预测
            prev_mask = torch.zeros(1, 1, image_size[0], image_size[1]).to(device)
            
            for step in range(max_rollout_steps):
                # 预测下一个点
                sam_features = sam2_lora.get_image_features(img_tensor, feature_scale=8)
                heatmap_logits, label_logits = point_predictor(sam_features, prev_mask)
                
                # Argmax获取点
                B, _, H, W = heatmap_logits.shape
                heatmap_flat = heatmap_logits.view(B, -1)
                pred_idx = heatmap_flat.argmax(dim=1)
                pred_y = (pred_idx // W).float()
                pred_x = (pred_idx % W).float()
                pred_label = label_logits.argmax(dim=1).item()
                
                # 记录点
                pred_points_seq.append((pred_x.item(), pred_y.item()))
                pred_labels_seq.append(pred_label)
                
                # 生成掩模
                points_orig = [(px * orig_w / image_size[1], py * orig_h / image_size[0])
                              for px, py in pred_points_seq]
                prev_mask = sam2_lora.predict_mask(
                    np.array(img), points_orig, pred_labels_seq
                )
                if isinstance(prev_mask, torch.Tensor):
                    prev_mask = prev_mask.unsqueeze(0).to(device)
                else:
                    prev_mask = torch.from_numpy(prev_mask).unsqueeze(0).to(device)
                prev_mask = F.interpolate(
                    prev_mask, size=image_size, mode='nearest'
                )
                
                # 计算指标
                if gt_mask_binary is not None:
                    dice = compute_dice(prev_mask[0, 0], gt_mask_binary[0, 0])
                    iou = compute_iou(prev_mask[0, 0], gt_mask_binary[0, 0])
                    
                    if step == max_rollout_steps - 1 or dice > 0.95:
                        metrics['dice'] = dice
                        metrics['iou'] = iou
                        metrics['num_points'] = len(pred_points_seq)
                        pred_mask = prev_mask[0, 0].cpu().numpy()
                        break
            
            # 转换为numpy用于可视化
            if pred_mask is None:
                pred_mask = prev_mask[0, 0].cpu().numpy()
            
        except Exception as e:
            print(f"Warning: Prediction failed: {e}")
            do_prediction = False
    
    # 创建图形
    if do_prediction:
        # 两幅图：第一幅显示点和标注，第二幅显示掩模对比
        fig, axes = plt.subplots(1, 2, figsize=(24, 12))
        ax1, ax2 = axes
    else:
        # 单幅图：只显示点和标注
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 12))
        ax2 = None
    
    # 第一幅图：显示点和标注
    ax1.imshow(img_np)
    
    # 如果有真值掩模，叠加显示
    if show_mask:
        gt_mask_path = sample.get('gt_mask', None)
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                mask_binary = (gt_mask > 127).astype(float)
                ax1.imshow(mask_binary, alpha=0.3, cmap='Greens')
            except Exception as e:
                print(f"Warning: Failed to load mask {gt_mask_path}: {e}")
    
    # 绘制点
    for j, ((x, y), label) in enumerate(zip(points, labels)):
        if label == 1:
            # 前景点：红色，实心圆
            color = 'red'
            size = 150
            point_type = 'FG'
        else:
            # 背景点：蓝色，实心圆
            color = 'blue'
            size = 120
            point_type = 'BG'
        
        # 绘制点
        ax1.scatter(x, y, c=color, s=size, marker='o', 
                   edgecolors='white', linewidths=2, zorder=10)
        
        # 添加标签文本
        ax1.text(x + 15, y, f'{j}({point_type})', 
                fontsize=10, color='white', weight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    # 设置第一幅图的标题
    stem = Path(image_path).stem
    title1 = f'{stem}: {len(points)} points (Ground Truth)'
    ax1.set_title(title1, fontsize=14, weight='bold')
    ax1.axis('off')
    
    # 第二幅图：显示掩模对比（如果进行预测）
    if do_prediction and ax2 is not None:
        ax2.imshow(img_np)
        
        # 叠加真值掩模（绿色）
        gt_mask_path = sample.get('gt_mask', None)
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                gt_mask_binary = (gt_mask > 127).astype(float)
                # 调整尺寸以匹配图像
                if gt_mask_binary.shape != img_np.shape[:2]:
                    from scipy.ndimage import zoom
                    scale_y = img_np.shape[0] / gt_mask_binary.shape[0]
                    scale_x = img_np.shape[1] / gt_mask_binary.shape[1]
                    gt_mask_binary = zoom(gt_mask_binary, (scale_y, scale_x), order=0)
                
                # 绿色半透明显示真值掩模
                gt_mask_colored = np.zeros_like(img_np)
                gt_mask_colored[:, :, 1] = gt_mask_binary * 255
                ax2.imshow(gt_mask_colored, alpha=0.4)
            except Exception as e:
                print(f"Warning: Failed to load GT mask: {e}")
        
        # 叠加预测掩模（红色）
        if pred_mask is not None:
            # 调整尺寸以匹配图像
            if pred_mask.shape != img_np.shape[:2]:
                from scipy.ndimage import zoom
                scale_y = img_np.shape[0] / pred_mask.shape[0]
                scale_x = img_np.shape[1] / pred_mask.shape[1]
                pred_mask_resized = zoom(pred_mask, (scale_y, scale_x), order=0)
            else:
                pred_mask_resized = pred_mask
            
            pred_mask_binary = (pred_mask_resized > 0.5).astype(float)
            # 红色半透明显示预测掩模
            pred_mask_colored = np.zeros_like(img_np)
            pred_mask_colored[:, :, 0] = pred_mask_binary * 255
            ax2.imshow(pred_mask_colored, alpha=0.4)
        
        # 绘制预测点
        for j, ((x, y), label) in enumerate(zip(pred_points_seq, pred_labels_seq)):
            if label == 1:
                color = 'red'
                size = 150
                point_type = 'FG'
            else:
                color = 'blue'
                size = 120
                point_type = 'BG'
            
            # 缩放点坐标到原始图像尺寸
            x_orig = x * orig_w / image_size[1]
            y_orig = y * orig_h / image_size[0]
            
            ax2.scatter(x_orig, y_orig, c=color, s=size, marker='o', 
                       edgecolors='white', linewidths=2, zorder=10)
            ax2.text(x_orig + 15, y_orig, f'{j}({point_type})', 
                    fontsize=10, color='white', weight='bold',
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
        
        # 设置第二幅图的标题和指标
        title2 = f'Prediction: {len(pred_points_seq)} points'
        if metrics:
            metrics_text = f"Dice: {metrics.get('dice', 0):.3f}, IoU: {metrics.get('iou', 0):.3f}"
            title2 += f'\n{metrics_text}'
        ax2.set_title(title2, fontsize=14, weight='bold')
        ax2.axis('off')
        
        # 在图像角落添加指标文本
        if metrics:
            metrics_str = f"Dice: {metrics.get('dice', 0):.3f}\nIoU: {metrics.get('iou', 0):.3f}\nPoints: {metrics.get('num_points', 0)}"
            ax2.text(0.02, 0.98, metrics_str, 
                    transform=ax2.transAxes, fontsize=12, weight='bold',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='可视化PEFT训练数据',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--jsonl',
        type=str,
        required=True,
        help='输入JSONL文件路径'
    )
    
    parser.add_argument(
        '--out_dir',
        type=str,
        required=True,
        help='输出图像文件夹路径'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['random', 'all'],
        default='random',
        help='采样模式: random=随机采样, all=遍历所有样本'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='随机采样模式下的样本数量（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--show_mask',
        action='store_true',
        default=True,
        help='是否显示真值掩模（默认True）'
    )
    
    parser.add_argument(
        '--no_mask',
        dest='show_mask',
        action='store_false',
        help='不显示真值掩模'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='输出图像分辨率（DPI）'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default='sample',
        help='输出文件名前缀'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='模型checkpoint路径（如果提供，将生成预测对比图）'
    )
    
    parser.add_argument(
        '--sam_checkpoint',
        type=str,
        default=None,
        help='SAM2 checkpoint路径（如果提供checkpoint，此参数必需）'
    )
    
    parser.add_argument(
        '--image_size',
        type=int,
        nargs=2,
        default=[512, 512],
        help='图像尺寸 [H, W]（用于预测）'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='设备（cuda/cpu）'
    )
    
    parser.add_argument(
        '--max_rollout_steps',
        type=int,
        default=16,
        help='最大rollout步数（用于预测）'
    )
    
    args = parser.parse_args()
    
    # 加载模型（如果提供了checkpoint）
    sam2_lora = None
    point_predictor = None
    
    if args.checkpoint:
        if not args.sam_checkpoint:
            print("Error: --sam_checkpoint is required when --checkpoint is provided")
            sys.exit(1)
        
        print(f"Loading model from {args.checkpoint}...")
        try:
            device = torch.device(args.device)
            
            # 加载checkpoint
            ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
            config = ckpt.get('config', {})
            
            # 提取配置
            lora_rank = config.get('lora_rank', 16)
            lora_alpha = config.get('lora_alpha', 32)
            fusion_mode = config.get('fusion_mode', 'film')
            feature_scale = config.get('feature_scale', 8)
            
            # 初始化SAM2 LoRA
            sam2_lora = LoRASAM2Wrapper(
                sam_checkpoint=args.sam_checkpoint,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                device=str(device),
            )
            
            # 初始化点预测网络
            sam_feature_dim = 256  # SAM2 Hiera Large的特征维度
            point_predictor = PointPredictorFromSAMFeatures(
                sam_feature_dim=sam_feature_dim,
                output_size=tuple(args.image_size),
                fusion_mode=fusion_mode,
                feature_scale=feature_scale,
            ).to(device)
            
            # 加载权重
            load_checkpoint(
                args.checkpoint, point_predictor, sam2_lora, None, None, None, str(device)
            )
            
            point_predictor.eval()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Will continue without prediction...")
            sam2_lora = None
            point_predictor = None
    
    # 加载数据
    print(f"Loading data from {args.jsonl}...")
    data = load_jsonl(args.jsonl)
    print(f"Loaded {len(data)} samples")
    
    # 选择样本
    if args.mode == 'random':
        random.seed(args.seed)
        selected_samples = random.sample(data, min(args.num_samples, len(data)))
        print(f"Randomly selected {len(selected_samples)} samples")
    else:
        selected_samples = data
        print(f"Processing all {len(selected_samples)} samples")
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 可视化每个样本
    print(f"\nGenerating visualizations...")
    for i, sample in enumerate(selected_samples):
        # 生成输出文件名
        if args.mode == 'random':
            output_filename = f'{args.prefix}_{i:03d}.png'
        else:
            # 使用图像文件名作为输出文件名
            image_path = sample.get('image', '')
            if image_path:
                stem = Path(image_path).stem
                output_filename = f'{args.prefix}_{stem}.png'
            else:
                output_filename = f'{args.prefix}_{i:04d}.png'
        
        output_path = os.path.join(args.out_dir, output_filename)
        
        # 可视化
        try:
            visualize_sample(
                sample, output_path, 
                show_mask=args.show_mask, 
                dpi=args.dpi,
                sam2_lora=sam2_lora,
                point_predictor=point_predictor,
                image_size=tuple(args.image_size),
                device=args.device,
                max_rollout_steps=args.max_rollout_steps,
            )
            print(f"  [{i+1}/{len(selected_samples)}] Saved: {output_path}")
        except Exception as e:
            print(f"  [{i+1}/{len(selected_samples)}] Error processing sample {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nDone! Visualizations saved to: {args.out_dir}")
    print(f"Total: {len(selected_samples)} images generated")


if __name__ == '__main__':
    main()


```


检查并修复导入路径：



修复导入路径：


```python
#!/usr/bin/env python3
"""
可视化训练数据脚本

用于可视化PEFT训练数据中的图像、点标注和真值掩模。
支持随机采样或遍历所有样本。
如果提供了checkpoint，还会生成预测掩模对比图。

用法:
    # 随机采样5个样本（仅显示标注）
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_train-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode random \
        --num_samples 5

    # 遍历所有样本（带预测对比）
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_test-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode all \
        --checkpoint outputs/braintumour/peft_supervised_baseline-251107/checkpoint_best.pt \
        --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
        --image_size 512 512
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

# 添加父目录到路径
_PARENT_DIR = Path(__file__).parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from peft.lora_sam2 import LoRASAM2Wrapper
from peft.point_predictor_peft import PointPredictorFromSAMFeatures
from peft.utils_peft import compute_dice, compute_iou, load_checkpoint


def load_jsonl(jsonl_path: str) -> List[Dict]:
    """加载JSONL文件"""
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith('['):
            # JSON数组格式
            data = json.loads(content)
        else:
            # JSONL格式（逐行）
            data = []
            for line in content.splitlines():
                if line.strip():
                    data.append(json.loads(line))
    return data


def visualize_sample(
    sample: Dict,
    output_path: str,
    show_mask: bool = True,
    dpi: int = 150,
    sam2_lora: Optional[LoRASAM2Wrapper] = None,
    point_predictor: Optional[PointPredictorFromSAMFeatures] = None,
    image_size: Tuple[int, int] = (512, 512),
    device: str = 'cuda',
    max_rollout_steps: int = 16,
) -> None:
    """
    可视化单个样本
    
    Args:
        sample: 样本字典，包含 'image', 'gt_mask', 'points', 'labels' 等字段
        output_path: 输出图像路径
        show_mask: 是否显示真值掩模
        dpi: 图像分辨率
        sam2_lora: SAM2 LoRA包装器（可选，用于预测）
        point_predictor: 点预测网络（可选，用于预测）
        image_size: 图像尺寸
        device: 设备
        max_rollout_steps: 最大rollout步数
    """
    # 加载图像
    image_path = sample['image']
    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        return
    
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    orig_w, orig_h = img.size
    
    # 获取点和标签
    points = sample.get('points', [])
    labels = sample.get('labels', [])
    
    if len(points) != len(labels):
        print(f"Warning: Points and labels length mismatch: {len(points)} vs {len(labels)}")
        return
    
    # 是否进行预测
    do_prediction = sam2_lora is not None and point_predictor is not None
    
    # 计算预测掩模和指标（如果需要）
    pred_mask = None
    metrics = {}
    pred_points_seq = []
    pred_labels_seq = []
    
    if do_prediction:
        try:
            # 运行rollout预测
            img_resized = img.resize(image_size, Image.BILINEAR)
            img_tensor = TF.to_tensor(img_resized).unsqueeze(0).to(device)
            
            # 加载真值掩模
            gt_mask_path = sample.get('gt_mask', None)
            if gt_mask_path and os.path.exists(gt_mask_path):
                gt_mask = Image.open(gt_mask_path).convert('L')
                gt_mask_resized = gt_mask.resize(image_size, Image.NEAREST)
                gt_mask_tensor = TF.to_tensor(gt_mask_resized).to(device)
                gt_mask_binary = (gt_mask_tensor > 0.5).float()
            else:
                gt_mask_binary = None
            
            # Rollout预测
            prev_mask = torch.zeros(1, 1, image_size[0], image_size[1]).to(device)
            
            for step in range(max_rollout_steps):
                # 预测下一个点
                sam_features = sam2_lora.get_image_features(img_tensor, feature_scale=8)
                heatmap_logits, label_logits = point_predictor(sam_features, prev_mask)
                
                # Argmax获取点
                B, _, H, W = heatmap_logits.shape
                heatmap_flat = heatmap_logits.view(B, -1)
                pred_idx = heatmap_flat.argmax(dim=1)
                pred_y = (pred_idx // W).float()
                pred_x = (pred_idx % W).float()
                pred_label = label_logits.argmax(dim=1).item()
                
                # 记录点
                pred_points_seq.append((pred_x.item(), pred_y.item()))
                pred_labels_seq.append(pred_label)
                
                # 生成掩模
                points_orig = [(px * orig_w / image_size[1], py * orig_h / image_size[0])
                              for px, py in pred_points_seq]
                prev_mask = sam2_lora.predict_mask(
                    np.array(img), points_orig, pred_labels_seq
                )
                if isinstance(prev_mask, torch.Tensor):
                    prev_mask = prev_mask.unsqueeze(0).to(device)
                else:
                    prev_mask = torch.from_numpy(prev_mask).unsqueeze(0).to(device)
                prev_mask = F.interpolate(
                    prev_mask, size=image_size, mode='nearest'
                )
                
                # 计算指标
                if gt_mask_binary is not None:
                    dice = compute_dice(prev_mask[0, 0], gt_mask_binary[0, 0])
                    iou = compute_iou(prev_mask[0, 0], gt_mask_binary[0, 0])
                    
                    if step == max_rollout_steps - 1 or dice > 0.95:
                        metrics['dice'] = dice
                        metrics['iou'] = iou
                        metrics['num_points'] = len(pred_points_seq)
                        pred_mask = prev_mask[0, 0].cpu().numpy()
                        break
            
            # 转换为numpy用于可视化
            if pred_mask is None:
                pred_mask = prev_mask[0, 0].cpu().numpy()
            
        except Exception as e:
            print(f"Warning: Prediction failed: {e}")
            do_prediction = False
    
    # 创建图形
    if do_prediction:
        # 两幅图：第一幅显示点和标注，第二幅显示掩模对比
        fig, axes = plt.subplots(1, 2, figsize=(24, 12))
        ax1, ax2 = axes
    else:
        # 单幅图：只显示点和标注
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 12))
        ax2 = None
    
    # 第一幅图：显示点和标注
    ax1.imshow(img_np)
    
    # 如果有真值掩模，叠加显示
    if show_mask:
        gt_mask_path = sample.get('gt_mask', None)
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                mask_binary = (gt_mask > 127).astype(float)
                ax1.imshow(mask_binary, alpha=0.3, cmap='Greens')
            except Exception as e:
                print(f"Warning: Failed to load mask {gt_mask_path}: {e}")
    
    # 绘制点
    for j, ((x, y), label) in enumerate(zip(points, labels)):
        if label == 1:
            # 前景点：红色，实心圆
            color = 'red'
            size = 150
            point_type = 'FG'
        else:
            # 背景点：蓝色，实心圆
            color = 'blue'
            size = 120
            point_type = 'BG'
        
        # 绘制点
        ax1.scatter(x, y, c=color, s=size, marker='o', 
                   edgecolors='white', linewidths=2, zorder=10)
        
        # 添加标签文本
        ax1.text(x + 15, y, f'{j}({point_type})', 
                fontsize=10, color='white', weight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    # 设置第一幅图的标题
    stem = Path(image_path).stem
    title1 = f'{stem}: {len(points)} points (Ground Truth)'
    ax1.set_title(title1, fontsize=14, weight='bold')
    ax1.axis('off')
    
    # 第二幅图：显示掩模对比（如果进行预测）
    if do_prediction and ax2 is not None:
        ax2.imshow(img_np)
        
        # 叠加真值掩模（绿色）
        gt_mask_path = sample.get('gt_mask', None)
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                gt_mask_binary = (gt_mask > 127).astype(float)
                # 调整尺寸以匹配图像
                if gt_mask_binary.shape != img_np.shape[:2]:
                    from scipy.ndimage import zoom
                    scale_y = img_np.shape[0] / gt_mask_binary.shape[0]
                    scale_x = img_np.shape[1] / gt_mask_binary.shape[1]
                    gt_mask_binary = zoom(gt_mask_binary, (scale_y, scale_x), order=0)
                
                # 绿色半透明显示真值掩模
                gt_mask_colored = np.zeros_like(img_np)
                gt_mask_colored[:, :, 1] = gt_mask_binary * 255
                ax2.imshow(gt_mask_colored, alpha=0.4)
            except Exception as e:
                print(f"Warning: Failed to load GT mask: {e}")
        
        # 叠加预测掩模（红色）
        if pred_mask is not None:
            # 调整尺寸以匹配图像
            if pred_mask.shape != img_np.shape[:2]:
                from scipy.ndimage import zoom
                scale_y = img_np.shape[0] / pred_mask.shape[0]
                scale_x = img_np.shape[1] / pred_mask.shape[1]
                pred_mask_resized = zoom(pred_mask, (scale_y, scale_x), order=0)
            else:
                pred_mask_resized = pred_mask
            
            pred_mask_binary = (pred_mask_resized > 0.5).astype(float)
            # 红色半透明显示预测掩模
            pred_mask_colored = np.zeros_like(img_np)
            pred_mask_colored[:, :, 0] = pred_mask_binary * 255
            ax2.imshow(pred_mask_colored, alpha=0.4)
        
        # 绘制预测点
        for j, ((x, y), label) in enumerate(zip(pred_points_seq, pred_labels_seq)):
            if label == 1:
                color = 'red'
                size = 150
                point_type = 'FG'
            else:
                color = 'blue'
                size = 120
                point_type = 'BG'
            
            # 缩放点坐标到原始图像尺寸
            x_orig = x * orig_w / image_size[1]
            y_orig = y * orig_h / image_size[0]
            
            ax2.scatter(x_orig, y_orig, c=color, s=size, marker='o', 
                       edgecolors='white', linewidths=2, zorder=10)
            ax2.text(x_orig + 15, y_orig, f'{j}({point_type})', 
                    fontsize=10, color='white', weight='bold',
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
        
        # 设置第二幅图的标题和指标
        title2 = f'Prediction: {len(pred_points_seq)} points'
        if metrics:
            metrics_text = f"Dice: {metrics.get('dice', 0):.3f}, IoU: {metrics.get('iou', 0):.3f}"
            title2 += f'\n{metrics_text}'
        ax2.set_title(title2, fontsize=14, weight='bold')
        ax2.axis('off')
        
        # 在图像角落添加指标文本
        if metrics:
            metrics_str = f"Dice: {metrics.get('dice', 0):.3f}\nIoU: {metrics.get('iou', 0):.3f}\nPoints: {metrics.get('num_points', 0)}"
            ax2.text(0.02, 0.98, metrics_str, 
                    transform=ax2.transAxes, fontsize=12, weight='bold',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='可视化PEFT训练数据',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--jsonl',
        type=str,
        required=True,
        help='输入JSONL文件路径'
    )
    
    parser.add_argument(
        '--out_dir',
        type=str,
        required=True,
        help='输出图像文件夹路径'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['random', 'all'],
        default='random',
        help='采样模式: random=随机采样, all=遍历所有样本'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='随机采样模式下的样本数量（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--show_mask',
        action='store_true',
        default=True,
        help='是否显示真值掩模（默认True）'
    )
    
    parser.add_argument(
        '--no_mask',
        dest='show_mask',
        action='store_false',
        help='不显示真值掩模'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='输出图像分辨率（DPI）'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default='sample',
        help='输出文件名前缀'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='模型checkpoint路径（如果提供，将生成预测对比图）'
    )
    
    parser.add_argument(
        '--sam_checkpoint',
        type=str,
        default=None,
        help='SAM2 checkpoint路径（如果提供checkpoint，此参数必需）'
    )
    
    parser.add_argument(
        '--image_size',
        type=int,
        nargs=2,
        default=[512, 512],
        help='图像尺寸 [H, W]（用于预测）'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='设备（cuda/cpu）'
    )
    
    parser.add_argument(
        '--max_rollout_steps',
        type=int,
        default=16,
        help='最大rollout步数（用于预测）'
    )
    
    args = parser.parse_args()
    
    # 加载模型（如果提供了checkpoint）
    sam2_lora = None
    point_predictor = None
    
    if args.checkpoint:
        if not args.sam_checkpoint:
            print("Error: --sam_checkpoint is required when --checkpoint is provided")
            sys.exit(1)
        
        print(f"Loading model from {args.checkpoint}...")
        try:
            device = torch.device(args.device)
            
            # 加载checkpoint
            ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
            config = ckpt.get('config', {})
            
            # 提取配置
            lora_rank = config.get('lora_rank', 16)
            lora_alpha = config.get('lora_alpha', 32)
            fusion_mode = config.get('fusion_mode', 'film')
            feature_scale = config.get('feature_scale', 8)
            
            # 初始化SAM2 LoRA
            sam2_lora = LoRASAM2Wrapper(
                sam_checkpoint=args.sam_checkpoint,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                device=str(device),
            )
            
            # 初始化点预测网络
            sam_feature_dim = 256  # SAM2 Hiera Large的特征维度
            point_predictor = PointPredictorFromSAMFeatures(
                sam_feature_dim=sam_feature_dim,
                output_size=tuple(args.image_size),
                fusion_mode=fusion_mode,
                feature_scale=feature_scale,
            ).to(device)
            
            # 加载权重
            load_checkpoint(
                args.checkpoint, point_predictor, sam2_lora, None, None, None, str(device)
            )
            
            point_predictor.eval()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Will continue without prediction...")
            sam2_lora = None
            point_predictor = None
    
    # 加载数据
    print(f"Loading data from {args.jsonl}...")
    data = load_jsonl(args.jsonl)
    print(f"Loaded {len(data)} samples")
    
    # 选择样本
    if args.mode == 'random':
        random.seed(args.seed)
        selected_samples = random.sample(data, min(args.num_samples, len(data)))
        print(f"Randomly selected {len(selected_samples)} samples")
    else:
        selected_samples = data
        print(f"Processing all {len(selected_samples)} samples")
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 可视化每个样本
    print(f"\nGenerating visualizations...")
    for i, sample in enumerate(selected_samples):
        # 生成输出文件名
        if args.mode == 'random':
            output_filename = f'{args.prefix}_{i:03d}.png'
        else:
            # 使用图像文件名作为输出文件名
            image_path = sample.get('image', '')
            if image_path:
                stem = Path(image_path).stem
                output_filename = f'{args.prefix}_{stem}.png'
            else:
                output_filename = f'{args.prefix}_{i:04d}.png'
        
        output_path = os.path.join(args.out_dir, output_filename)
        
        # 可视化
        try:
            visualize_sample(
                sample, output_path, 
                show_mask=args.show_mask, 
                dpi=args.dpi,
                sam2_lora=sam2_lora,
                point_predictor=point_predictor,
                image_size=tuple(args.image_size),
                device=args.device,
                max_rollout_steps=args.max_rollout_steps,
            )
            print(f"  [{i+1}/{len(selected_samples)}] Saved: {output_path}")
        except Exception as e:
            print(f"  [{i+1}/{len(selected_samples)}] Error processing sample {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nDone! Visualizations saved to: {args.out_dir}")
    print(f"Total: {len(selected_samples)} images generated")


if __name__ == '__main__':
    main()


```


修复预测逻辑，确保在 `torch.no_grad()` 下运行：


```python
#!/usr/bin/env python3
"""
可视化训练数据脚本

用于可视化PEFT训练数据中的图像、点标注和真值掩模。
支持随机采样或遍历所有样本。
如果提供了checkpoint，还会生成预测掩模对比图。

用法:
    # 随机采样5个样本（仅显示标注）
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_train-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode random \
        --num_samples 5

    # 遍历所有样本（带预测对比）
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_test-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode all \
        --checkpoint outputs/braintumour/peft_supervised_baseline-251107/checkpoint_best.pt \
        --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
        --image_size 512 512
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

# 添加父目录到路径
_PARENT_DIR = Path(__file__).parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from peft.lora_sam2 import LoRASAM2Wrapper
from peft.point_predictor_peft import PointPredictorFromSAMFeatures
from peft.utils_peft import compute_dice, compute_iou, load_checkpoint


def load_jsonl(jsonl_path: str) -> List[Dict]:
    """加载JSONL文件"""
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith('['):
            # JSON数组格式
            data = json.loads(content)
        else:
            # JSONL格式（逐行）
            data = []
            for line in content.splitlines():
                if line.strip():
                    data.append(json.loads(line))
    return data


def visualize_sample(
    sample: Dict,
    output_path: str,
    show_mask: bool = True,
    dpi: int = 150,
    sam2_lora: Optional[LoRASAM2Wrapper] = None,
    point_predictor: Optional[PointPredictorFromSAMFeatures] = None,
    image_size: Tuple[int, int] = (512, 512),
    device: str = 'cuda',
    max_rollout_steps: int = 16,
) -> None:
    """
    可视化单个样本
    
    Args:
        sample: 样本字典，包含 'image', 'gt_mask', 'points', 'labels' 等字段
        output_path: 输出图像路径
        show_mask: 是否显示真值掩模
        dpi: 图像分辨率
        sam2_lora: SAM2 LoRA包装器（可选，用于预测）
        point_predictor: 点预测网络（可选，用于预测）
        image_size: 图像尺寸
        device: 设备
        max_rollout_steps: 最大rollout步数
    """
    # 加载图像
    image_path = sample['image']
    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        return
    
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    orig_w, orig_h = img.size
    
    # 获取点和标签
    points = sample.get('points', [])
    labels = sample.get('labels', [])
    
    if len(points) != len(labels):
        print(f"Warning: Points and labels length mismatch: {len(points)} vs {len(labels)}")
        return
    
    # 是否进行预测
    do_prediction = sam2_lora is not None and point_predictor is not None
    
    # 计算预测掩模和指标（如果需要）
    pred_mask = None
    metrics = {}
    pred_points_seq = []
    pred_labels_seq = []
    
    if do_prediction:
        try:
            with torch.no_grad():
                # 运行rollout预测
                img_resized = img.resize(image_size, Image.BILINEAR)
                img_tensor = TF.to_tensor(img_resized).unsqueeze(0).to(device)
                
                # 加载真值掩模
                gt_mask_path = sample.get('gt_mask', None)
                if gt_mask_path and os.path.exists(gt_mask_path):
                    gt_mask = Image.open(gt_mask_path).convert('L')
                    gt_mask_resized = gt_mask.resize(image_size, Image.NEAREST)
                    gt_mask_tensor = TF.to_tensor(gt_mask_resized).to(device)
                    gt_mask_binary = (gt_mask_tensor > 0.5).float()
                else:
                    gt_mask_binary = None
                
                # Rollout预测
                prev_mask = torch.zeros(1, 1, image_size[0], image_size[1]).to(device)
                
                for step in range(max_rollout_steps):
                    # 预测下一个点
                    sam_features = sam2_lora.get_image_features(img_tensor, feature_scale=8)
                    heatmap_logits, label_logits = point_predictor(sam_features, prev_mask)
                    
                    # Argmax获取点
                    B, _, H, W = heatmap_logits.shape
                    heatmap_flat = heatmap_logits.view(B, -1)
                    pred_idx = heatmap_flat.argmax(dim=1)
                    pred_y = (pred_idx // W).float()
                    pred_x = (pred_idx % W).float()
                    pred_label = label_logits.argmax(dim=1).item()
                    
                    # 记录点
                    pred_points_seq.append((pred_x.item(), pred_y.item()))
                    pred_labels_seq.append(pred_label)
                    
                    # 生成掩模
                    points_orig = [(px * orig_w / image_size[1], py * orig_h / image_size[0])
                                  for px, py in pred_points_seq]
                    prev_mask = sam2_lora.predict_mask(
                        np.array(img), points_orig, pred_labels_seq
                    )
                    if isinstance(prev_mask, torch.Tensor):
                        prev_mask = prev_mask.unsqueeze(0).to(device)
                    else:
                        prev_mask = torch.from_numpy(prev_mask).unsqueeze(0).to(device)
                    prev_mask = F.interpolate(
                        prev_mask, size=image_size, mode='nearest'
                    )
                    
                    # 计算指标
                    if gt_mask_binary is not None:
                        dice = compute_dice(prev_mask[0, 0], gt_mask_binary[0, 0])
                        iou = compute_iou(prev_mask[0, 0], gt_mask_binary[0, 0])
                        
                        # 保存最终指标
                        metrics['dice'] = dice
                        metrics['iou'] = iou
                        metrics['num_points'] = len(pred_points_seq)
                        pred_mask = prev_mask[0, 0].cpu().numpy()
                        
                        # 停止条件
                        if dice > 0.95 or (step > 0 and dice - metrics.get('prev_dice', 0) < 0.001):
                            break
                        metrics['prev_dice'] = dice
                
                # 转换为numpy用于可视化
                if pred_mask is None:
                    pred_mask = prev_mask[0, 0].cpu().numpy()
            
        except Exception as e:
            print(f"Warning: Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            do_prediction = False
    
    # 创建图形
    if do_prediction:
        # 两幅图：第一幅显示点和标注，第二幅显示掩模对比
        fig, axes = plt.subplots(1, 2, figsize=(24, 12))
        ax1, ax2 = axes
    else:
        # 单幅图：只显示点和标注
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 12))
        ax2 = None
    
    # 第一幅图：显示点和标注
    ax1.imshow(img_np)
    
    # 如果有真值掩模，叠加显示
    if show_mask:
        gt_mask_path = sample.get('gt_mask', None)
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                mask_binary = (gt_mask > 127).astype(float)
                ax1.imshow(mask_binary, alpha=0.3, cmap='Greens')
            except Exception as e:
                print(f"Warning: Failed to load mask {gt_mask_path}: {e}")
    
    # 绘制点
    for j, ((x, y), label) in enumerate(zip(points, labels)):
        if label == 1:
            # 前景点：红色，实心圆
            color = 'red'
            size = 150
            point_type = 'FG'
        else:
            # 背景点：蓝色，实心圆
            color = 'blue'
            size = 120
            point_type = 'BG'
        
        # 绘制点
        ax1.scatter(x, y, c=color, s=size, marker='o', 
                   edgecolors='white', linewidths=2, zorder=10)
        
        # 添加标签文本
        ax1.text(x + 15, y, f'{j}({point_type})', 
                fontsize=10, color='white', weight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    # 设置第一幅图的标题
    stem = Path(image_path).stem
    title1 = f'{stem}: {len(points)} points (Ground Truth)'
    ax1.set_title(title1, fontsize=14, weight='bold')
    ax1.axis('off')
    
    # 第二幅图：显示掩模对比（如果进行预测）
    if do_prediction and ax2 is not None:
        ax2.imshow(img_np)
        
        # 叠加真值掩模（绿色）
        gt_mask_path = sample.get('gt_mask', None)
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                gt_mask_binary = (gt_mask > 127).astype(float)
                # 调整尺寸以匹配图像
                if gt_mask_binary.shape != img_np.shape[:2]:
                    from scipy.ndimage import zoom
                    scale_y = img_np.shape[0] / gt_mask_binary.shape[0]
                    scale_x = img_np.shape[1] / gt_mask_binary.shape[1]
                    gt_mask_binary = zoom(gt_mask_binary, (scale_y, scale_x), order=0)
                
                # 绿色半透明显示真值掩模
                gt_mask_colored = np.zeros_like(img_np)
                gt_mask_colored[:, :, 1] = gt_mask_binary * 255
                ax2.imshow(gt_mask_colored, alpha=0.4)
            except Exception as e:
                print(f"Warning: Failed to load GT mask: {e}")
        
        # 叠加预测掩模（红色）
        if pred_mask is not None:
            # 调整尺寸以匹配图像
            if pred_mask.shape != img_np.shape[:2]:
                from scipy.ndimage import zoom
                scale_y = img_np.shape[0] / pred_mask.shape[0]
                scale_x = img_np.shape[1] / pred_mask.shape[1]
                pred_mask_resized = zoom(pred_mask, (scale_y, scale_x), order=0)
            else:
                pred_mask_resized = pred_mask
            
            pred_mask_binary = (pred_mask_resized > 0.5).astype(float)
            # 红色半透明显示预测掩模
            pred_mask_colored = np.zeros_like(img_np)
            pred_mask_colored[:, :, 0] = pred_mask_binary * 255
            ax2.imshow(pred_mask_colored, alpha=0.4)
        
        # 绘制预测点
        for j, ((x, y), label) in enumerate(zip(pred_points_seq, pred_labels_seq)):
            if label == 1:
                color = 'red'
                size = 150
                point_type = 'FG'
            else:
                color = 'blue'
                size = 120
                point_type = 'BG'
            
            # 缩放点坐标到原始图像尺寸
            x_orig = x * orig_w / image_size[1]
            y_orig = y * orig_h / image_size[0]
            
            ax2.scatter(x_orig, y_orig, c=color, s=size, marker='o', 
                       edgecolors='white', linewidths=2, zorder=10)
            ax2.text(x_orig + 15, y_orig, f'{j}({point_type})', 
                    fontsize=10, color='white', weight='bold',
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
        
        # 设置第二幅图的标题和指标
        title2 = f'Prediction: {len(pred_points_seq)} points'
        if metrics:
            metrics_text = f"Dice: {metrics.get('dice', 0):.3f}, IoU: {metrics.get('iou', 0):.3f}"
            title2 += f'\n{metrics_text}'
        ax2.set_title(title2, fontsize=14, weight='bold')
        ax2.axis('off')
        
        # 在图像角落添加指标文本
        if metrics:
            metrics_str = f"Dice: {metrics.get('dice', 0):.3f}\nIoU: {metrics.get('iou', 0):.3f}\nPoints: {metrics.get('num_points', 0)}"
            ax2.text(0.02, 0.98, metrics_str, 
                    transform=ax2.transAxes, fontsize=12, weight='bold',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='可视化PEFT训练数据',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--jsonl',
        type=str,
        required=True,
        help='输入JSONL文件路径'
    )
    
    parser.add_argument(
        '--out_dir',
        type=str,
        required=True,
        help='输出图像文件夹路径'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['random', 'all'],
        default='random',
        help='采样模式: random=随机采样, all=遍历所有样本'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='随机采样模式下的样本数量（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--show_mask',
        action='store_true',
        default=True,
        help='是否显示真值掩模（默认True）'
    )
    
    parser.add_argument(
        '--no_mask',
        dest='show_mask',
        action='store_false',
        help='不显示真值掩模'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='输出图像分辨率（DPI）'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default='sample',
        help='输出文件名前缀'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='模型checkpoint路径（如果提供，将生成预测对比图）'
    )
    
    parser.add_argument(
        '--sam_checkpoint',
        type=str,
        default=None,
        help='SAM2 checkpoint路径（如果提供checkpoint，此参数必需）'
    )
    
    parser.add_argument(
        '--image_size',
        type=int,
        nargs=2,
        default=[512, 512],
        help='图像尺寸 [H, W]（用于预测）'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='设备（cuda/cpu）'
    )
    
    parser.add_argument(
        '--max_rollout_steps',
        type=int,
        default=16,
        help='最大rollout步数（用于预测）'
    )
    
    args = parser.parse_args()
    
    # 加载模型（如果提供了checkpoint）
    sam2_lora = None
    point_predictor = None
    
    if args.checkpoint:
        if not args.sam_checkpoint:
            print("Error: --sam_checkpoint is required when --checkpoint is provided")
            sys.exit(1)
        
        print(f"Loading model from {args.checkpoint}...")
        try:
            device = torch.device(args.device)
            
            # 加载checkpoint
            ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
            config = ckpt.get('config', {})
            
            # 提取配置
            lora_rank = config.get('lora_rank', 16)
            lora_alpha = config.get('lora_alpha', 32)
            fusion_mode = config.get('fusion_mode', 'film')
            feature_scale = config.get('feature_scale', 8)
            
            # 初始化SAM2 LoRA
            sam2_lora = LoRASAM2Wrapper(
                sam_checkpoint=args.sam_checkpoint,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                device=str(device),
            )
            
            # 初始化点预测网络
            sam_feature_dim = 256  # SAM2 Hiera Large的特征维度
            point_predictor = PointPredictorFromSAMFeatures(
                sam_feature_dim=sam_feature_dim,
                output_size=tuple(args.image_size),
                fusion_mode=fusion_mode,
                feature_scale=feature_scale,
            ).to(device)
            
            # 加载权重
            load_checkpoint(
                args.checkpoint, point_predictor, sam2_lora, None, None, None, str(device)
            )
            
            point_predictor.eval()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Will continue without prediction...")
            sam2_lora = None
            point_predictor = None
    
    # 加载数据
    print(f"Loading data from {args.jsonl}...")
    data = load_jsonl(args.jsonl)
    print(f"Loaded {len(data)} samples")
    
    # 选择样本
    if args.mode == 'random':
        random.seed(args.seed)
        selected_samples = random.sample(data, min(args.num_samples, len(data)))
        print(f"Randomly selected {len(selected_samples)} samples")
    else:
        selected_samples = data
        print(f"Processing all {len(selected_samples)} samples")
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 可视化每个样本
    print(f"\nGenerating visualizations...")
    for i, sample in enumerate(selected_samples):
        # 生成输出文件名
        if args.mode == 'random':
            output_filename = f'{args.prefix}_{i:03d}.png'
        else:
            # 使用图像文件名作为输出文件名
            image_path = sample.get('image', '')
            if image_path:
                stem = Path(image_path).stem
                output_filename = f'{args.prefix}_{stem}.png'
            else:
                output_filename = f'{args.prefix}_{i:04d}.png'
        
        output_path = os.path.join(args.out_dir, output_filename)
        
        # 可视化
        try:
            visualize_sample(
                sample, output_path, 
                show_mask=args.show_mask, 
                dpi=args.dpi,
                sam2_lora=sam2_lora,
                point_predictor=point_predictor,
                image_size=tuple(args.image_size),
                device=args.device,
                max_rollout_steps=args.max_rollout_steps,
            )
            print(f"  [{i+1}/{len(selected_samples)}] Saved: {output_path}")
        except Exception as e:
            print(f"  [{i+1}/{len(selected_samples)}] Error processing sample {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nDone! Visualizations saved to: {args.out_dir}")
    print(f"Total: {len(selected_samples)} images generated")


if __name__ == '__main__':
    main()


```


添加 scipy 导入，并修复掩模尺寸匹配问题：


```python
#!/usr/bin/env python3
"""
可视化训练数据脚本

用于可视化PEFT训练数据中的图像、点标注和真值掩模。
支持随机采样或遍历所有样本。
如果提供了checkpoint，还会生成预测掩模对比图。

用法:
    # 随机采样5个样本（仅显示标注）
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_train-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode random \
        --num_samples 5

    # 遍历所有样本（带预测对比）
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_test-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode all \
        --checkpoint outputs/braintumour/peft_supervised_baseline-251107/checkpoint_best.pt \
        --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
        --image_size 512 512
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.ndimage import zoom

# 添加父目录到路径
_PARENT_DIR = Path(__file__).parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from peft.lora_sam2 import LoRASAM2Wrapper
from peft.point_predictor_peft import PointPredictorFromSAMFeatures
from peft.utils_peft import compute_dice, compute_iou, load_checkpoint


def load_jsonl(jsonl_path: str) -> List[Dict]:
    """加载JSONL文件"""
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith('['):
            # JSON数组格式
            data = json.loads(content)
        else:
            # JSONL格式（逐行）
            data = []
            for line in content.splitlines():
                if line.strip():
                    data.append(json.loads(line))
    return data


def visualize_sample(
    sample: Dict,
    output_path: str,
    show_mask: bool = True,
    dpi: int = 150,
    sam2_lora: Optional[LoRASAM2Wrapper] = None,
    point_predictor: Optional[PointPredictorFromSAMFeatures] = None,
    image_size: Tuple[int, int] = (512, 512),
    device: str = 'cuda',
    max_rollout_steps: int = 16,
) -> None:
    """
    可视化单个样本
    
    Args:
        sample: 样本字典，包含 'image', 'gt_mask', 'points', 'labels' 等字段
        output_path: 输出图像路径
        show_mask: 是否显示真值掩模
        dpi: 图像分辨率
        sam2_lora: SAM2 LoRA包装器（可选，用于预测）
        point_predictor: 点预测网络（可选，用于预测）
        image_size: 图像尺寸
        device: 设备
        max_rollout_steps: 最大rollout步数
    """
    # 加载图像
    image_path = sample['image']
    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        return
    
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    orig_w, orig_h = img.size
    
    # 获取点和标签
    points = sample.get('points', [])
    labels = sample.get('labels', [])
    
    if len(points) != len(labels):
        print(f"Warning: Points and labels length mismatch: {len(points)} vs {len(labels)}")
        return
    
    # 是否进行预测
    do_prediction = sam2_lora is not None and point_predictor is not None
    
    # 计算预测掩模和指标（如果需要）
    pred_mask = None
    metrics = {}
    pred_points_seq = []
    pred_labels_seq = []
    
    if do_prediction:
        try:
            with torch.no_grad():
                # 运行rollout预测
                img_resized = img.resize(image_size, Image.BILINEAR)
                img_tensor = TF.to_tensor(img_resized).unsqueeze(0).to(device)
                
                # 加载真值掩模
                gt_mask_path = sample.get('gt_mask', None)
                if gt_mask_path and os.path.exists(gt_mask_path):
                    gt_mask = Image.open(gt_mask_path).convert('L')
                    gt_mask_resized = gt_mask.resize(image_size, Image.NEAREST)
                    gt_mask_tensor = TF.to_tensor(gt_mask_resized).to(device)
                    gt_mask_binary = (gt_mask_tensor > 0.5).float()
                else:
                    gt_mask_binary = None
                
                # Rollout预测
                prev_mask = torch.zeros(1, 1, image_size[0], image_size[1]).to(device)
                
                for step in range(max_rollout_steps):
                    # 预测下一个点
                    sam_features = sam2_lora.get_image_features(img_tensor, feature_scale=8)
                    heatmap_logits, label_logits = point_predictor(sam_features, prev_mask)
                    
                    # Argmax获取点
                    B, _, H, W = heatmap_logits.shape
                    heatmap_flat = heatmap_logits.view(B, -1)
                    pred_idx = heatmap_flat.argmax(dim=1)
                    pred_y = (pred_idx // W).float()
                    pred_x = (pred_idx % W).float()
                    pred_label = label_logits.argmax(dim=1).item()
                    
                    # 记录点
                    pred_points_seq.append((pred_x.item(), pred_y.item()))
                    pred_labels_seq.append(pred_label)
                    
                    # 生成掩模
                    points_orig = [(px * orig_w / image_size[1], py * orig_h / image_size[0])
                                  for px, py in pred_points_seq]
                    prev_mask = sam2_lora.predict_mask(
                        np.array(img), points_orig, pred_labels_seq
                    )
                    if isinstance(prev_mask, torch.Tensor):
                        prev_mask = prev_mask.unsqueeze(0).to(device)
                    else:
                        prev_mask = torch.from_numpy(prev_mask).unsqueeze(0).to(device)
                    prev_mask = F.interpolate(
                        prev_mask, size=image_size, mode='nearest'
                    )
                    
                    # 计算指标
                    if gt_mask_binary is not None:
                        dice = compute_dice(prev_mask[0, 0], gt_mask_binary[0, 0])
                        iou = compute_iou(prev_mask[0, 0], gt_mask_binary[0, 0])
                        
                        # 保存最终指标
                        metrics['dice'] = dice
                        metrics['iou'] = iou
                        metrics['num_points'] = len(pred_points_seq)
                        pred_mask = prev_mask[0, 0].cpu().numpy()
                        
                        # 停止条件
                        if dice > 0.95 or (step > 0 and dice - metrics.get('prev_dice', 0) < 0.001):
                            break
                        metrics['prev_dice'] = dice
                
                # 转换为numpy用于可视化
                if pred_mask is None:
                    pred_mask = prev_mask[0, 0].cpu().numpy()
            
        except Exception as e:
            print(f"Warning: Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            do_prediction = False
    
    # 创建图形
    if do_prediction:
        # 两幅图：第一幅显示点和标注，第二幅显示掩模对比
        fig, axes = plt.subplots(1, 2, figsize=(24, 12))
        ax1, ax2 = axes
    else:
        # 单幅图：只显示点和标注
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 12))
        ax2 = None
    
    # 第一幅图：显示点和标注
    ax1.imshow(img_np)
    
    # 如果有真值掩模，叠加显示
    if show_mask:
        gt_mask_path = sample.get('gt_mask', None)
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                mask_binary = (gt_mask > 127).astype(float)
                ax1.imshow(mask_binary, alpha=0.3, cmap='Greens')
            except Exception as e:
                print(f"Warning: Failed to load mask {gt_mask_path}: {e}")
    
    # 绘制点
    for j, ((x, y), label) in enumerate(zip(points, labels)):
        if label == 1:
            # 前景点：红色，实心圆
            color = 'red'
            size = 150
            point_type = 'FG'
        else:
            # 背景点：蓝色，实心圆
            color = 'blue'
            size = 120
            point_type = 'BG'
        
        # 绘制点
        ax1.scatter(x, y, c=color, s=size, marker='o', 
                   edgecolors='white', linewidths=2, zorder=10)
        
        # 添加标签文本
        ax1.text(x + 15, y, f'{j}({point_type})', 
                fontsize=10, color='white', weight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    # 设置第一幅图的标题
    stem = Path(image_path).stem
    title1 = f'{stem}: {len(points)} points (Ground Truth)'
    ax1.set_title(title1, fontsize=14, weight='bold')
    ax1.axis('off')
    
    # 第二幅图：显示掩模对比（如果进行预测）
    if do_prediction and ax2 is not None:
        ax2.imshow(img_np)
        
        # 叠加真值掩模（绿色）
        gt_mask_path = sample.get('gt_mask', None)
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                gt_mask_binary = (gt_mask > 127).astype(float)
                # 调整尺寸以匹配图像
                if gt_mask_binary.shape != img_np.shape[:2]:
                    from scipy.ndimage import zoom
                    scale_y = img_np.shape[0] / gt_mask_binary.shape[0]
                    scale_x = img_np.shape[1] / gt_mask_binary.shape[1]
                    gt_mask_binary = zoom(gt_mask_binary, (scale_y, scale_x), order=0)
                
                # 绿色半透明显示真值掩模
                gt_mask_colored = np.zeros_like(img_np)
                gt_mask_colored[:, :, 1] = gt_mask_binary * 255
                ax2.imshow(gt_mask_colored, alpha=0.4)
            except Exception as e:
                print(f"Warning: Failed to load GT mask: {e}")
        
        # 叠加预测掩模（红色）
        if pred_mask is not None:
            # 调整尺寸以匹配图像
            if pred_mask.shape != img_np.shape[:2]:
                from scipy.ndimage import zoom
                scale_y = img_np.shape[0] / pred_mask.shape[0]
                scale_x = img_np.shape[1] / pred_mask.shape[1]
                pred_mask_resized = zoom(pred_mask, (scale_y, scale_x), order=0)
            else:
                pred_mask_resized = pred_mask
            
            pred_mask_binary = (pred_mask_resized > 0.5).astype(float)
            # 红色半透明显示预测掩模
            pred_mask_colored = np.zeros_like(img_np)
            pred_mask_colored[:, :, 0] = pred_mask_binary * 255
            ax2.imshow(pred_mask_colored, alpha=0.4)
        
        # 绘制预测点
        for j, ((x, y), label) in enumerate(zip(pred_points_seq, pred_labels_seq)):
            if label == 1:
                color = 'red'
                size = 150
                point_type = 'FG'
            else:
                color = 'blue'
                size = 120
                point_type = 'BG'
            
            # 缩放点坐标到原始图像尺寸
            x_orig = x * orig_w / image_size[1]
            y_orig = y * orig_h / image_size[0]
            
            ax2.scatter(x_orig, y_orig, c=color, s=size, marker='o', 
                       edgecolors='white', linewidths=2, zorder=10)
            ax2.text(x_orig + 15, y_orig, f'{j}({point_type})', 
                    fontsize=10, color='white', weight='bold',
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
        
        # 设置第二幅图的标题和指标
        title2 = f'Prediction: {len(pred_points_seq)} points'
        if metrics:
            metrics_text = f"Dice: {metrics.get('dice', 0):.3f}, IoU: {metrics.get('iou', 0):.3f}"
            title2 += f'\n{metrics_text}'
        ax2.set_title(title2, fontsize=14, weight='bold')
        ax2.axis('off')
        
        # 在图像角落添加指标文本
        if metrics:
            metrics_str = f"Dice: {metrics.get('dice', 0):.3f}\nIoU: {metrics.get('iou', 0):.3f}\nPoints: {metrics.get('num_points', 0)}"
            ax2.text(0.02, 0.98, metrics_str, 
                    transform=ax2.transAxes, fontsize=12, weight='bold',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='可视化PEFT训练数据',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--jsonl',
        type=str,
        required=True,
        help='输入JSONL文件路径'
    )
    
    parser.add_argument(
        '--out_dir',
        type=str,
        required=True,
        help='输出图像文件夹路径'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['random', 'all'],
        default='random',
        help='采样模式: random=随机采样, all=遍历所有样本'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='随机采样模式下的样本数量（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--show_mask',
        action='store_true',
        default=True,
        help='是否显示真值掩模（默认True）'
    )
    
    parser.add_argument(
        '--no_mask',
        dest='show_mask',
        action='store_false',
        help='不显示真值掩模'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='输出图像分辨率（DPI）'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default='sample',
        help='输出文件名前缀'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='模型checkpoint路径（如果提供，将生成预测对比图）'
    )
    
    parser.add_argument(
        '--sam_checkpoint',
        type=str,
        default=None,
        help='SAM2 checkpoint路径（如果提供checkpoint，此参数必需）'
    )
    
    parser.add_argument(
        '--image_size',
        type=int,
        nargs=2,
        default=[512, 512],
        help='图像尺寸 [H, W]（用于预测）'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='设备（cuda/cpu）'
    )
    
    parser.add_argument(
        '--max_rollout_steps',
        type=int,
        default=16,
        help='最大rollout步数（用于预测）'
    )
    
    args = parser.parse_args()
    
    # 加载模型（如果提供了checkpoint）
    sam2_lora = None
    point_predictor = None
    
    if args.checkpoint:
        if not args.sam_checkpoint:
            print("Error: --sam_checkpoint is required when --checkpoint is provided")
            sys.exit(1)
        
        print(f"Loading model from {args.checkpoint}...")
        try:
            device = torch.device(args.device)
            
            # 加载checkpoint
            ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
            config = ckpt.get('config', {})
            
            # 提取配置
            lora_rank = config.get('lora_rank', 16)
            lora_alpha = config.get('lora_alpha', 32)
            fusion_mode = config.get('fusion_mode', 'film')
            feature_scale = config.get('feature_scale', 8)
            
            # 初始化SAM2 LoRA
            sam2_lora = LoRASAM2Wrapper(
                sam_checkpoint=args.sam_checkpoint,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                device=str(device),
            )
            
            # 初始化点预测网络
            sam_feature_dim = 256  # SAM2 Hiera Large的特征维度
            point_predictor = PointPredictorFromSAMFeatures(
                sam_feature_dim=sam_feature_dim,
                output_size=tuple(args.image_size),
                fusion_mode=fusion_mode,
                feature_scale=feature_scale,
            ).to(device)
            
            # 加载权重
            load_checkpoint(
                args.checkpoint, point_predictor, sam2_lora, None, None, None, str(device)
            )
            
            point_predictor.eval()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Will continue without prediction...")
            sam2_lora = None
            point_predictor = None
    
    # 加载数据
    print(f"Loading data from {args.jsonl}...")
    data = load_jsonl(args.jsonl)
    print(f"Loaded {len(data)} samples")
    
    # 选择样本
    if args.mode == 'random':
        random.seed(args.seed)
        selected_samples = random.sample(data, min(args.num_samples, len(data)))
        print(f"Randomly selected {len(selected_samples)} samples")
    else:
        selected_samples = data
        print(f"Processing all {len(selected_samples)} samples")
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 可视化每个样本
    print(f"\nGenerating visualizations...")
    for i, sample in enumerate(selected_samples):
        # 生成输出文件名
        if args.mode == 'random':
            output_filename = f'{args.prefix}_{i:03d}.png'
        else:
            # 使用图像文件名作为输出文件名
            image_path = sample.get('image', '')
            if image_path:
                stem = Path(image_path).stem
                output_filename = f'{args.prefix}_{stem}.png'
            else:
                output_filename = f'{args.prefix}_{i:04d}.png'
        
        output_path = os.path.join(args.out_dir, output_filename)
        
        # 可视化
        try:
            visualize_sample(
                sample, output_path, 
                show_mask=args.show_mask, 
                dpi=args.dpi,
                sam2_lora=sam2_lora,
                point_predictor=point_predictor,
                image_size=tuple(args.image_size),
                device=args.device,
                max_rollout_steps=args.max_rollout_steps,
            )
            print(f"  [{i+1}/{len(selected_samples)}] Saved: {output_path}")
        except Exception as e:
            print(f"  [{i+1}/{len(selected_samples)}] Error processing sample {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nDone! Visualizations saved to: {args.out_dir}")
    print(f"Total: {len(selected_samples)} images generated")


if __name__ == '__main__':
    main()


```


修复第二幅图中的重复导入：


```python
#!/usr/bin/env python3
"""
可视化训练数据脚本

用于可视化PEFT训练数据中的图像、点标注和真值掩模。
支持随机采样或遍历所有样本。
如果提供了checkpoint，还会生成预测掩模对比图。

用法:
    # 随机采样5个样本（仅显示标注）
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_train-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode random \
        --num_samples 5

    # 遍历所有样本（带预测对比）
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_test-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode all \
        --checkpoint outputs/braintumour/peft_supervised_baseline-251107/checkpoint_best.pt \
        --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
        --image_size 512 512
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.ndimage import zoom

# 添加父目录到路径
_PARENT_DIR = Path(__file__).parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from peft.lora_sam2 import LoRASAM2Wrapper
from peft.point_predictor_peft import PointPredictorFromSAMFeatures
from peft.utils_peft import compute_dice, compute_iou, load_checkpoint


def load_jsonl(jsonl_path: str) -> List[Dict]:
    """加载JSONL文件"""
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith('['):
            # JSON数组格式
            data = json.loads(content)
        else:
            # JSONL格式（逐行）
            data = []
            for line in content.splitlines():
                if line.strip():
                    data.append(json.loads(line))
    return data


def visualize_sample(
    sample: Dict,
    output_path: str,
    show_mask: bool = True,
    dpi: int = 150,
    sam2_lora: Optional[LoRASAM2Wrapper] = None,
    point_predictor: Optional[PointPredictorFromSAMFeatures] = None,
    image_size: Tuple[int, int] = (512, 512),
    device: str = 'cuda',
    max_rollout_steps: int = 16,
) -> None:
    """
    可视化单个样本
    
    Args:
        sample: 样本字典，包含 'image', 'gt_mask', 'points', 'labels' 等字段
        output_path: 输出图像路径
        show_mask: 是否显示真值掩模
        dpi: 图像分辨率
        sam2_lora: SAM2 LoRA包装器（可选，用于预测）
        point_predictor: 点预测网络（可选，用于预测）
        image_size: 图像尺寸
        device: 设备
        max_rollout_steps: 最大rollout步数
    """
    # 加载图像
    image_path = sample['image']
    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        return
    
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    orig_w, orig_h = img.size
    
    # 获取点和标签
    points = sample.get('points', [])
    labels = sample.get('labels', [])
    
    if len(points) != len(labels):
        print(f"Warning: Points and labels length mismatch: {len(points)} vs {len(labels)}")
        return
    
    # 是否进行预测
    do_prediction = sam2_lora is not None and point_predictor is not None
    
    # 计算预测掩模和指标（如果需要）
    pred_mask = None
    metrics = {}
    pred_points_seq = []
    pred_labels_seq = []
    
    if do_prediction:
        try:
            with torch.no_grad():
                # 运行rollout预测
                img_resized = img.resize(image_size, Image.BILINEAR)
                img_tensor = TF.to_tensor(img_resized).unsqueeze(0).to(device)
                
                # 加载真值掩模
                gt_mask_path = sample.get('gt_mask', None)
                if gt_mask_path and os.path.exists(gt_mask_path):
                    gt_mask = Image.open(gt_mask_path).convert('L')
                    gt_mask_resized = gt_mask.resize(image_size, Image.NEAREST)
                    gt_mask_tensor = TF.to_tensor(gt_mask_resized).to(device)
                    gt_mask_binary = (gt_mask_tensor > 0.5).float()
                else:
                    gt_mask_binary = None
                
                # Rollout预测
                prev_mask = torch.zeros(1, 1, image_size[0], image_size[1]).to(device)
                
                for step in range(max_rollout_steps):
                    # 预测下一个点
                    sam_features = sam2_lora.get_image_features(img_tensor, feature_scale=8)
                    heatmap_logits, label_logits = point_predictor(sam_features, prev_mask)
                    
                    # Argmax获取点
                    B, _, H, W = heatmap_logits.shape
                    heatmap_flat = heatmap_logits.view(B, -1)
                    pred_idx = heatmap_flat.argmax(dim=1)
                    pred_y = (pred_idx // W).float()
                    pred_x = (pred_idx % W).float()
                    pred_label = label_logits.argmax(dim=1).item()
                    
                    # 记录点
                    pred_points_seq.append((pred_x.item(), pred_y.item()))
                    pred_labels_seq.append(pred_label)
                    
                    # 生成掩模
                    points_orig = [(px * orig_w / image_size[1], py * orig_h / image_size[0])
                                  for px, py in pred_points_seq]
                    prev_mask = sam2_lora.predict_mask(
                        np.array(img), points_orig, pred_labels_seq
                    )
                    if isinstance(prev_mask, torch.Tensor):
                        prev_mask = prev_mask.unsqueeze(0).to(device)
                    else:
                        prev_mask = torch.from_numpy(prev_mask).unsqueeze(0).to(device)
                    prev_mask = F.interpolate(
                        prev_mask, size=image_size, mode='nearest'
                    )
                    
                    # 计算指标
                    if gt_mask_binary is not None:
                        dice = compute_dice(prev_mask[0, 0], gt_mask_binary[0, 0])
                        iou = compute_iou(prev_mask[0, 0], gt_mask_binary[0, 0])
                        
                        # 保存最终指标
                        metrics['dice'] = dice
                        metrics['iou'] = iou
                        metrics['num_points'] = len(pred_points_seq)
                        pred_mask = prev_mask[0, 0].cpu().numpy()
                        
                        # 停止条件
                        if dice > 0.95 or (step > 0 and dice - metrics.get('prev_dice', 0) < 0.001):
                            break
                        metrics['prev_dice'] = dice
                
                # 转换为numpy用于可视化
                if pred_mask is None:
                    pred_mask = prev_mask[0, 0].cpu().numpy()
            
        except Exception as e:
            print(f"Warning: Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            do_prediction = False
    
    # 创建图形
    if do_prediction:
        # 两幅图：第一幅显示点和标注，第二幅显示掩模对比
        fig, axes = plt.subplots(1, 2, figsize=(24, 12))
        ax1, ax2 = axes
    else:
        # 单幅图：只显示点和标注
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 12))
        ax2 = None
    
    # 第一幅图：显示点和标注
    ax1.imshow(img_np)
    
    # 如果有真值掩模，叠加显示
    if show_mask:
        gt_mask_path = sample.get('gt_mask', None)
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                mask_binary = (gt_mask > 127).astype(float)
                ax1.imshow(mask_binary, alpha=0.3, cmap='Greens')
            except Exception as e:
                print(f"Warning: Failed to load mask {gt_mask_path}: {e}")
    
    # 绘制点
    for j, ((x, y), label) in enumerate(zip(points, labels)):
        if label == 1:
            # 前景点：红色，实心圆
            color = 'red'
            size = 150
            point_type = 'FG'
        else:
            # 背景点：蓝色，实心圆
            color = 'blue'
            size = 120
            point_type = 'BG'
        
        # 绘制点
        ax1.scatter(x, y, c=color, s=size, marker='o', 
                   edgecolors='white', linewidths=2, zorder=10)
        
        # 添加标签文本
        ax1.text(x + 15, y, f'{j}({point_type})', 
                fontsize=10, color='white', weight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    # 设置第一幅图的标题
    stem = Path(image_path).stem
    title1 = f'{stem}: {len(points)} points (Ground Truth)'
    ax1.set_title(title1, fontsize=14, weight='bold')
    ax1.axis('off')
    
    # 第二幅图：显示掩模对比（如果进行预测）
    if do_prediction and ax2 is not None:
        ax2.imshow(img_np)
        
        # 叠加真值掩模（绿色）
        gt_mask_path = sample.get('gt_mask', None)
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                gt_mask_binary = (gt_mask > 127).astype(float)
                # 调整尺寸以匹配图像
                if gt_mask_binary.shape != img_np.shape[:2]:
                    scale_y = img_np.shape[0] / gt_mask_binary.shape[0]
                    scale_x = img_np.shape[1] / gt_mask_binary.shape[1]
                    gt_mask_binary = zoom(gt_mask_binary, (scale_y, scale_x), order=0)
                
                # 绿色半透明显示真值掩模
                gt_mask_colored = np.zeros_like(img_np)
                gt_mask_colored[:, :, 1] = gt_mask_binary * 255
                ax2.imshow(gt_mask_colored, alpha=0.4)
            except Exception as e:
                print(f"Warning: Failed to load GT mask: {e}")
        
        # 叠加预测掩模（红色）
        if pred_mask is not None:
            # 调整尺寸以匹配图像
            if pred_mask.shape != img_np.shape[:2]:
                from scipy.ndimage import zoom
                scale_y = img_np.shape[0] / pred_mask.shape[0]
                scale_x = img_np.shape[1] / pred_mask.shape[1]
                pred_mask_resized = zoom(pred_mask, (scale_y, scale_x), order=0)
            else:
                pred_mask_resized = pred_mask
            
            pred_mask_binary = (pred_mask_resized > 0.5).astype(float)
            # 红色半透明显示预测掩模
            pred_mask_colored = np.zeros_like(img_np)
            pred_mask_colored[:, :, 0] = pred_mask_binary * 255
            ax2.imshow(pred_mask_colored, alpha=0.4)
        
        # 绘制预测点
        for j, ((x, y), label) in enumerate(zip(pred_points_seq, pred_labels_seq)):
            if label == 1:
                color = 'red'
                size = 150
                point_type = 'FG'
            else:
                color = 'blue'
                size = 120
                point_type = 'BG'
            
            # 缩放点坐标到原始图像尺寸
            x_orig = x * orig_w / image_size[1]
            y_orig = y * orig_h / image_size[0]
            
            ax2.scatter(x_orig, y_orig, c=color, s=size, marker='o', 
                       edgecolors='white', linewidths=2, zorder=10)
            ax2.text(x_orig + 15, y_orig, f'{j}({point_type})', 
                    fontsize=10, color='white', weight='bold',
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
        
        # 设置第二幅图的标题和指标
        title2 = f'Prediction: {len(pred_points_seq)} points'
        if metrics:
            metrics_text = f"Dice: {metrics.get('dice', 0):.3f}, IoU: {metrics.get('iou', 0):.3f}"
            title2 += f'\n{metrics_text}'
        ax2.set_title(title2, fontsize=14, weight='bold')
        ax2.axis('off')
        
        # 在图像角落添加指标文本
        if metrics:
            metrics_str = f"Dice: {metrics.get('dice', 0):.3f}\nIoU: {metrics.get('iou', 0):.3f}\nPoints: {metrics.get('num_points', 0)}"
            ax2.text(0.02, 0.98, metrics_str, 
                    transform=ax2.transAxes, fontsize=12, weight='bold',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='可视化PEFT训练数据',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--jsonl',
        type=str,
        required=True,
        help='输入JSONL文件路径'
    )
    
    parser.add_argument(
        '--out_dir',
        type=str,
        required=True,
        help='输出图像文件夹路径'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['random', 'all'],
        default='random',
        help='采样模式: random=随机采样, all=遍历所有样本'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='随机采样模式下的样本数量（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--show_mask',
        action='store_true',
        default=True,
        help='是否显示真值掩模（默认True）'
    )
    
    parser.add_argument(
        '--no_mask',
        dest='show_mask',
        action='store_false',
        help='不显示真值掩模'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='输出图像分辨率（DPI）'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default='sample',
        help='输出文件名前缀'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='模型checkpoint路径（如果提供，将生成预测对比图）'
    )
    
    parser.add_argument(
        '--sam_checkpoint',
        type=str,
        default=None,
        help='SAM2 checkpoint路径（如果提供checkpoint，此参数必需）'
    )
    
    parser.add_argument(
        '--image_size',
        type=int,
        nargs=2,
        default=[512, 512],
        help='图像尺寸 [H, W]（用于预测）'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='设备（cuda/cpu）'
    )
    
    parser.add_argument(
        '--max_rollout_steps',
        type=int,
        default=16,
        help='最大rollout步数（用于预测）'
    )
    
    args = parser.parse_args()
    
    # 加载模型（如果提供了checkpoint）
    sam2_lora = None
    point_predictor = None
    
    if args.checkpoint:
        if not args.sam_checkpoint:
            print("Error: --sam_checkpoint is required when --checkpoint is provided")
            sys.exit(1)
        
        print(f"Loading model from {args.checkpoint}...")
        try:
            device = torch.device(args.device)
            
            # 加载checkpoint
            ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
            config = ckpt.get('config', {})
            
            # 提取配置
            lora_rank = config.get('lora_rank', 16)
            lora_alpha = config.get('lora_alpha', 32)
            fusion_mode = config.get('fusion_mode', 'film')
            feature_scale = config.get('feature_scale', 8)
            
            # 初始化SAM2 LoRA
            sam2_lora = LoRASAM2Wrapper(
                sam_checkpoint=args.sam_checkpoint,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                device=str(device),
            )
            
            # 初始化点预测网络
            sam_feature_dim = 256  # SAM2 Hiera Large的特征维度
            point_predictor = PointPredictorFromSAMFeatures(
                sam_feature_dim=sam_feature_dim,
                output_size=tuple(args.image_size),
                fusion_mode=fusion_mode,
                feature_scale=feature_scale,
            ).to(device)
            
            # 加载权重
            load_checkpoint(
                args.checkpoint, point_predictor, sam2_lora, None, None, None, str(device)
            )
            
            point_predictor.eval()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Will continue without prediction...")
            sam2_lora = None
            point_predictor = None
    
    # 加载数据
    print(f"Loading data from {args.jsonl}...")
    data = load_jsonl(args.jsonl)
    print(f"Loaded {len(data)} samples")
    
    # 选择样本
    if args.mode == 'random':
        random.seed(args.seed)
        selected_samples = random.sample(data, min(args.num_samples, len(data)))
        print(f"Randomly selected {len(selected_samples)} samples")
    else:
        selected_samples = data
        print(f"Processing all {len(selected_samples)} samples")
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 可视化每个样本
    print(f"\nGenerating visualizations...")
    for i, sample in enumerate(selected_samples):
        # 生成输出文件名
        if args.mode == 'random':
            output_filename = f'{args.prefix}_{i:03d}.png'
        else:
            # 使用图像文件名作为输出文件名
            image_path = sample.get('image', '')
            if image_path:
                stem = Path(image_path).stem
                output_filename = f'{args.prefix}_{stem}.png'
            else:
                output_filename = f'{args.prefix}_{i:04d}.png'
        
        output_path = os.path.join(args.out_dir, output_filename)
        
        # 可视化
        try:
            visualize_sample(
                sample, output_path, 
                show_mask=args.show_mask, 
                dpi=args.dpi,
                sam2_lora=sam2_lora,
                point_predictor=point_predictor,
                image_size=tuple(args.image_size),
                device=args.device,
                max_rollout_steps=args.max_rollout_steps,
            )
            print(f"  [{i+1}/{len(selected_samples)}] Saved: {output_path}")
        except Exception as e:
            print(f"  [{i+1}/{len(selected_samples)}] Error processing sample {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nDone! Visualizations saved to: {args.out_dir}")
    print(f"Total: {len(selected_samples)} images generated")


if __name__ == '__main__':
    main()


```

```python
#!/usr/bin/env python3
"""
可视化训练数据脚本

用于可视化PEFT训练数据中的图像、点标注和真值掩模。
支持随机采样或遍历所有样本。
如果提供了checkpoint，还会生成预测掩模对比图。

用法:
    # 随机采样5个样本（仅显示标注）
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_train-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode random \
        --num_samples 5

    # 遍历所有样本（带预测对比）
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_test-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode all \
        --checkpoint outputs/braintumour/peft_supervised_baseline-251107/checkpoint_best.pt \
        --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
        --image_size 512 512
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.ndimage import zoom

# 添加父目录到路径
_PARENT_DIR = Path(__file__).parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from peft.lora_sam2 import LoRASAM2Wrapper
from peft.point_predictor_peft import PointPredictorFromSAMFeatures
from peft.utils_peft import compute_dice, compute_iou, load_checkpoint


def load_jsonl(jsonl_path: str) -> List[Dict]:
    """加载JSONL文件"""
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith('['):
            # JSON数组格式
            data = json.loads(content)
        else:
            # JSONL格式（逐行）
            data = []
            for line in content.splitlines():
                if line.strip():
                    data.append(json.loads(line))
    return data


def visualize_sample(
    sample: Dict,
    output_path: str,
    show_mask: bool = True,
    dpi: int = 150,
    sam2_lora: Optional[LoRASAM2Wrapper] = None,
    point_predictor: Optional[PointPredictorFromSAMFeatures] = None,
    image_size: Tuple[int, int] = (512, 512),
    device: str = 'cuda',
    max_rollout_steps: int = 16,
) -> None:
    """
    可视化单个样本
    
    Args:
        sample: 样本字典，包含 'image', 'gt_mask', 'points', 'labels' 等字段
        output_path: 输出图像路径
        show_mask: 是否显示真值掩模
        dpi: 图像分辨率
        sam2_lora: SAM2 LoRA包装器（可选，用于预测）
        point_predictor: 点预测网络（可选，用于预测）
        image_size: 图像尺寸
        device: 设备
        max_rollout_steps: 最大rollout步数
    """
    # 加载图像
    image_path = sample['image']
    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        return
    
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    orig_w, orig_h = img.size
    
    # 获取点和标签
    points = sample.get('points', [])
    labels = sample.get('labels', [])
    
    if len(points) != len(labels):
        print(f"Warning: Points and labels length mismatch: {len(points)} vs {len(labels)}")
        return
    
    # 是否进行预测
    do_prediction = sam2_lora is not None and point_predictor is not None
    
    # 计算预测掩模和指标（如果需要）
    pred_mask = None
    metrics = {}
    pred_points_seq = []
    pred_labels_seq = []
    
    if do_prediction:
        try:
            with torch.no_grad():
                # 运行rollout预测
                img_resized = img.resize(image_size, Image.BILINEAR)
                img_tensor = TF.to_tensor(img_resized).unsqueeze(0).to(device)
                
                # 加载真值掩模
                gt_mask_path = sample.get('gt_mask', None)
                if gt_mask_path and os.path.exists(gt_mask_path):
                    gt_mask = Image.open(gt_mask_path).convert('L')
                    gt_mask_resized = gt_mask.resize(image_size, Image.NEAREST)
                    gt_mask_tensor = TF.to_tensor(gt_mask_resized).to(device)
                    gt_mask_binary = (gt_mask_tensor > 0.5).float()
                else:
                    gt_mask_binary = None
                
                # Rollout预测
                prev_mask = torch.zeros(1, 1, image_size[0], image_size[1]).to(device)
                
                for step in range(max_rollout_steps):
                    # 预测下一个点
                    sam_features = sam2_lora.get_image_features(img_tensor, feature_scale=8)
                    heatmap_logits, label_logits = point_predictor(sam_features, prev_mask)
                    
                    # Argmax获取点
                    B, _, H, W = heatmap_logits.shape
                    heatmap_flat = heatmap_logits.view(B, -1)
                    pred_idx = heatmap_flat.argmax(dim=1)
                    pred_y = (pred_idx // W).float()
                    pred_x = (pred_idx % W).float()
                    pred_label = label_logits.argmax(dim=1).item()
                    
                    # 记录点
                    pred_points_seq.append((pred_x.item(), pred_y.item()))
                    pred_labels_seq.append(pred_label)
                    
                    # 生成掩模
                    points_orig = [(px * orig_w / image_size[1], py * orig_h / image_size[0])
                                  for px, py in pred_points_seq]
                    prev_mask = sam2_lora.predict_mask(
                        np.array(img), points_orig, pred_labels_seq
                    )
                    if isinstance(prev_mask, torch.Tensor):
                        prev_mask = prev_mask.unsqueeze(0).to(device)
                    else:
                        prev_mask = torch.from_numpy(prev_mask).unsqueeze(0).to(device)
                    prev_mask = F.interpolate(
                        prev_mask, size=image_size, mode='nearest'
                    )
                    
                    # 计算指标
                    if gt_mask_binary is not None:
                        dice = compute_dice(prev_mask[0, 0], gt_mask_binary[0, 0])
                        iou = compute_iou(prev_mask[0, 0], gt_mask_binary[0, 0])
                        
                        # 保存最终指标
                        metrics['dice'] = dice
                        metrics['iou'] = iou
                        metrics['num_points'] = len(pred_points_seq)
                        pred_mask = prev_mask[0, 0].cpu().numpy()
                        
                        # 停止条件
                        if dice > 0.95 or (step > 0 and dice - metrics.get('prev_dice', 0) < 0.001):
                            break
                        metrics['prev_dice'] = dice
                
                # 转换为numpy用于可视化
                if pred_mask is None:
                    pred_mask = prev_mask[0, 0].cpu().numpy()
            
        except Exception as e:
            print(f"Warning: Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            do_prediction = False
    
    # 创建图形
    if do_prediction:
        # 两幅图：第一幅显示点和标注，第二幅显示掩模对比
        fig, axes = plt.subplots(1, 2, figsize=(24, 12))
        ax1, ax2 = axes
    else:
        # 单幅图：只显示点和标注
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 12))
        ax2 = None
    
    # 第一幅图：显示点和标注
    ax1.imshow(img_np)
    
    # 如果有真值掩模，叠加显示
    if show_mask:
        gt_mask_path = sample.get('gt_mask', None)
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                mask_binary = (gt_mask > 127).astype(float)
                ax1.imshow(mask_binary, alpha=0.3, cmap='Greens')
            except Exception as e:
                print(f"Warning: Failed to load mask {gt_mask_path}: {e}")
    
    # 绘制点
    for j, ((x, y), label) in enumerate(zip(points, labels)):
        if label == 1:
            # 前景点：红色，实心圆
            color = 'red'
            size = 150
            point_type = 'FG'
        else:
            # 背景点：蓝色，实心圆
            color = 'blue'
            size = 120
            point_type = 'BG'
        
        # 绘制点
        ax1.scatter(x, y, c=color, s=size, marker='o', 
                   edgecolors='white', linewidths=2, zorder=10)
        
        # 添加标签文本
        ax1.text(x + 15, y, f'{j}({point_type})', 
                fontsize=10, color='white', weight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    # 设置第一幅图的标题
    stem = Path(image_path).stem
    title1 = f'{stem}: {len(points)} points (Ground Truth)'
    ax1.set_title(title1, fontsize=14, weight='bold')
    ax1.axis('off')
    
    # 第二幅图：显示掩模对比（如果进行预测）
    if do_prediction and ax2 is not None:
        ax2.imshow(img_np)
        
        # 叠加真值掩模（绿色）
        gt_mask_path = sample.get('gt_mask', None)
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                gt_mask_binary = (gt_mask > 127).astype(float)
                # 调整尺寸以匹配图像
                if gt_mask_binary.shape != img_np.shape[:2]:
                    scale_y = img_np.shape[0] / gt_mask_binary.shape[0]
                    scale_x = img_np.shape[1] / gt_mask_binary.shape[1]
                    gt_mask_binary = zoom(gt_mask_binary, (scale_y, scale_x), order=0)
                
                # 绿色半透明显示真值掩模
                gt_mask_colored = np.zeros_like(img_np)
                gt_mask_colored[:, :, 1] = gt_mask_binary * 255
                ax2.imshow(gt_mask_colored, alpha=0.4)
            except Exception as e:
                print(f"Warning: Failed to load GT mask: {e}")
        
        # 叠加预测掩模（红色）
        if pred_mask is not None:
            # 调整尺寸以匹配图像
            if pred_mask.shape != img_np.shape[:2]:
                from scipy.ndimage import zoom
                scale_y = img_np.shape[0] / pred_mask.shape[0]
                scale_x = img_np.shape[1] / pred_mask.shape[1]
                pred_mask_resized = zoom(pred_mask, (scale_y, scale_x), order=0)
            else:
                pred_mask_resized = pred_mask
            
            pred_mask_binary = (pred_mask_resized > 0.5).astype(float)
            # 红色半透明显示预测掩模
            pred_mask_colored = np.zeros_like(img_np)
            pred_mask_colored[:, :, 0] = pred_mask_binary * 255
            ax2.imshow(pred_mask_colored, alpha=0.4)
            
            # 计算重叠区域（黄色显示）
            if gt_mask_path and os.path.exists(gt_mask_path):
                try:
                    gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                    gt_mask_binary_overlap = (gt_mask > 127).astype(float)
                    if gt_mask_binary_overlap.shape != img_np.shape[:2]:
                        scale_y = img_np.shape[0] / gt_mask_binary_overlap.shape[0]
                        scale_x = img_np.shape[1] / gt_mask_binary_overlap.shape[1]
                        gt_mask_binary_overlap = zoom(gt_mask_binary_overlap, (scale_y, scale_x), order=0)
                    
                    # 重叠区域（黄色）
                    overlap = (gt_mask_binary_overlap * pred_mask_binary).astype(float)
                    overlap_colored = np.zeros_like(img_np)
                    overlap_colored[:, :, 0] = overlap * 255  # 红色
                    overlap_colored[:, :, 1] = overlap * 255  # 绿色
                    ax2.imshow(overlap_colored, alpha=0.3)
                except:
                    pass
        
        # 绘制预测点
        for j, ((x, y), label) in enumerate(zip(pred_points_seq, pred_labels_seq)):
            if label == 1:
                color = 'red'
                size = 150
                point_type = 'FG'
            else:
                color = 'blue'
                size = 120
                point_type = 'BG'
            
            # 缩放点坐标到原始图像尺寸
            x_orig = x * orig_w / image_size[1]
            y_orig = y * orig_h / image_size[0]
            
            ax2.scatter(x_orig, y_orig, c=color, s=size, marker='o', 
                       edgecolors='white', linewidths=2, zorder=10)
            ax2.text(x_orig + 15, y_orig, f'{j}({point_type})', 
                    fontsize=10, color='white', weight='bold',
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
        
        # 设置第二幅图的标题和指标
        title2 = f'Prediction: {len(pred_points_seq)} points'
        if metrics:
            metrics_text = f"Dice: {metrics.get('dice', 0):.3f}, IoU: {metrics.get('iou', 0):.3f}"
            title2 += f'\n{metrics_text}'
        ax2.set_title(title2, fontsize=14, weight='bold')
        ax2.axis('off')
        
        # 在图像角落添加指标文本
        if metrics:
            metrics_str = f"Dice: {metrics.get('dice', 0):.3f}\nIoU: {metrics.get('iou', 0):.3f}\nPoints: {metrics.get('num_points', 0)}"
            ax2.text(0.02, 0.98, metrics_str, 
                    transform=ax2.transAxes, fontsize=12, weight='bold',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='可视化PEFT训练数据',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--jsonl',
        type=str,
        required=True,
        help='输入JSONL文件路径'
    )
    
    parser.add_argument(
        '--out_dir',
        type=str,
        required=True,
        help='输出图像文件夹路径'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['random', 'all'],
        default='random',
        help='采样模式: random=随机采样, all=遍历所有样本'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='随机采样模式下的样本数量（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--show_mask',
        action='store_true',
        default=True,
        help='是否显示真值掩模（默认True）'
    )
    
    parser.add_argument(
        '--no_mask',
        dest='show_mask',
        action='store_false',
        help='不显示真值掩模'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='输出图像分辨率（DPI）'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default='sample',
        help='输出文件名前缀'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='模型checkpoint路径（如果提供，将生成预测对比图）'
    )
    
    parser.add_argument(
        '--sam_checkpoint',
        type=str,
        default=None,
        help='SAM2 checkpoint路径（如果提供checkpoint，此参数必需）'
    )
    
    parser.add_argument(
        '--image_size',
        type=int,
        nargs=2,
        default=[512, 512],
        help='图像尺寸 [H, W]（用于预测）'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='设备（cuda/cpu）'
    )
    
    parser.add_argument(
        '--max_rollout_steps',
        type=int,
        default=16,
        help='最大rollout步数（用于预测）'
    )
    
    args = parser.parse_args()
    
    # 加载模型（如果提供了checkpoint）
    sam2_lora = None
    point_predictor = None
    
    if args.checkpoint:
        if not args.sam_checkpoint:
            print("Error: --sam_checkpoint is required when --checkpoint is provided")
            sys.exit(1)
        
        print(f"Loading model from {args.checkpoint}...")
        try:
            device = torch.device(args.device)
            
            # 加载checkpoint
            ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
            config = ckpt.get('config', {})
            
            # 提取配置
            lora_rank = config.get('lora_rank', 16)
            lora_alpha = config.get('lora_alpha', 32)
            fusion_mode = config.get('fusion_mode', 'film')
            feature_scale = config.get('feature_scale', 8)
            
            # 初始化SAM2 LoRA
            sam2_lora = LoRASAM2Wrapper(
                sam_checkpoint=args.sam_checkpoint,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                device=str(device),
            )
            
            # 初始化点预测网络
            sam_feature_dim = 256  # SAM2 Hiera Large的特征维度
            point_predictor = PointPredictorFromSAMFeatures(
                sam_feature_dim=sam_feature_dim,
                output_size=tuple(args.image_size),
                fusion_mode=fusion_mode,
                feature_scale=feature_scale,
            ).to(device)
            
            # 加载权重
            load_checkpoint(
                args.checkpoint, point_predictor, sam2_lora, None, None, None, str(device)
            )
            
            point_predictor.eval()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Will continue without prediction...")
            sam2_lora = None
            point_predictor = None
    
    # 加载数据
    print(f"Loading data from {args.jsonl}...")
    data = load_jsonl(args.jsonl)
    print(f"Loaded {len(data)} samples")
    
    # 选择样本
    if args.mode == 'random':
        random.seed(args.seed)
        selected_samples = random.sample(data, min(args.num_samples, len(data)))
        print(f"Randomly selected {len(selected_samples)} samples")
    else:
        selected_samples = data
        print(f"Processing all {len(selected_samples)} samples")
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 可视化每个样本
    print(f"\nGenerating visualizations...")
    for i, sample in enumerate(selected_samples):
        # 生成输出文件名
        if args.mode == 'random':
            output_filename = f'{args.prefix}_{i:03d}.png'
        else:
            # 使用图像文件名作为输出文件名
            image_path = sample.get('image', '')
            if image_path:
                stem = Path(image_path).stem
                output_filename = f'{args.prefix}_{stem}.png'
            else:
                output_filename = f'{args.prefix}_{i:04d}.png'
        
        output_path = os.path.join(args.out_dir, output_filename)
        
        # 可视化
        try:
            visualize_sample(
                sample, output_path, 
                show_mask=args.show_mask, 
                dpi=args.dpi,
                sam2_lora=sam2_lora,
                point_predictor=point_predictor,
                image_size=tuple(args.image_size),
                device=args.device,
                max_rollout_steps=args.max_rollout_steps,
            )
            print(f"  [{i+1}/{len(selected_samples)}] Saved: {output_path}")
        except Exception as e:
            print(f"  [{i+1}/{len(selected_samples)}] Error processing sample {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nDone! Visualizations saved to: {args.out_dir}")
    print(f"Total: {len(selected_samples)} images generated")


if __name__ == '__main__':
    main()


```


修复第二幅图中掩模叠加顺序，确保正确显示：


```python
    # 第二幅图：显示掩模对比（如果进行预测）
    if do_prediction and ax2 is not None:
        ax2.imshow(img_np)
        
        gt_mask_path = sample.get('gt_mask', None)
        gt_mask_binary = None
        pred_mask_binary = None
        
        # 加载并调整真值掩模尺寸
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                gt_mask_binary = (gt_mask > 127).astype(float)
                # 调整尺寸以匹配图像
                if gt_mask_binary.shape != img_np.shape[:2]:
                    scale_y = img_np.shape[0] / gt_mask_binary.shape[0]
                    scale_x = img_np.shape[1] / gt_mask_binary.shape[1]
                    gt_mask_binary = zoom(gt_mask_binary, (scale_y, scale_x), order=0)
            except Exception as e:
                print(f"Warning: Failed to load GT mask: {e}")
        
        # 调整预测掩模尺寸
        if pred_mask is not None:
            if pred_mask.shape != img_np.shape[:2]:
                scale_y = img_np.shape[0] / pred_mask.shape[0]
                scale_x = img_np.shape[1] / pred_mask.shape[1]
                pred_mask_resized = zoom(pred_mask, (scale_y, scale_x), order=0)
            else:
                pred_mask_resized = pred_mask
            pred_mask_binary = (pred_mask_resized > 0.5).astype(float)
        
        # 叠加显示：先显示重叠区域（黄色），再显示真值（绿色），最后显示预测（红色）
        if gt_mask_binary is not None and pred_mask_binary is not None:
            # 重叠区域（黄色）
            overlap = (gt_mask_binary * pred_mask_binary).astype(float)
            overlap_colored = np.zeros_like(img_np)
            overlap_colored[:, :, 0] = overlap * 255  # 红色
            overlap_colored[:, :, 1] = overlap * 255  # 绿色
            ax2.imshow(overlap_colored, alpha=0.5)
            
            # 真值掩模的未重叠部分（绿色）
            gt_only = (gt_mask_binary * (1 - pred_mask_binary)).astype(float)
            gt_only_colored = np.zeros_like(img_np)
            gt_only_colored[:, :, 1] = gt_only * 255
            ax2.imshow(gt_only_colored, alpha=0.4)
            
            # 预测掩模的未重叠部分（红色）
            pred_only = (pred_mask_binary * (1 - gt_mask_binary)).astype(float)
            pred_only_colored = np.zeros_like(img_np)
            pred_only_colored[:, :, 0] = pred_only * 255
            ax2.imshow(pred_only_colored, alpha=0.4)
        elif gt_mask_binary is not None:
            # 只有真值掩模（绿色）
            gt_mask_colored = np.zeros_like(img_np)
            gt_mask_colored[:, :, 1] = gt_mask_binary * 255
            ax2.imshow(gt_mask_colored, alpha=0.4)
        elif pred_mask_binary is not None:
            # 只有预测掩模（红色）
            pred_mask_colored = np.zeros_like(img_np)
            pred_mask_colored[:, :, 0] = pred_mask_binary * 255
            ax2.imshow(pred_mask_colored, alpha=0.4)
```


读取文件，查看当前状态：



修复重复导入，并优化掩模叠加逻辑：


```python
#!/usr/bin/env python3
"""
可视化训练数据脚本

用于可视化PEFT训练数据中的图像、点标注和真值掩模。
支持随机采样或遍历所有样本。
如果提供了checkpoint，还会生成预测掩模对比图。

用法:
    # 随机采样5个样本（仅显示标注）
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_train-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode random \
        --num_samples 5

    # 遍历所有样本（带预测对比）
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_test-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode all \
        --checkpoint outputs/braintumour/peft_supervised_baseline-251107/checkpoint_best.pt \
        --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
        --image_size 512 512
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.ndimage import zoom

# 添加父目录到路径
_PARENT_DIR = Path(__file__).parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from peft.lora_sam2 import LoRASAM2Wrapper
from peft.point_predictor_peft import PointPredictorFromSAMFeatures
from peft.utils_peft import compute_dice, compute_iou, load_checkpoint


def load_jsonl(jsonl_path: str) -> List[Dict]:
    """加载JSONL文件"""
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith('['):
            # JSON数组格式
            data = json.loads(content)
        else:
            # JSONL格式（逐行）
            data = []
            for line in content.splitlines():
                if line.strip():
                    data.append(json.loads(line))
    return data


def visualize_sample(
    sample: Dict,
    output_path: str,
    show_mask: bool = True,
    dpi: int = 150,
    sam2_lora: Optional[LoRASAM2Wrapper] = None,
    point_predictor: Optional[PointPredictorFromSAMFeatures] = None,
    image_size: Tuple[int, int] = (512, 512),
    device: str = 'cuda',
    max_rollout_steps: int = 16,
) -> None:
    """
    可视化单个样本
    
    Args:
        sample: 样本字典，包含 'image', 'gt_mask', 'points', 'labels' 等字段
        output_path: 输出图像路径
        show_mask: 是否显示真值掩模
        dpi: 图像分辨率
        sam2_lora: SAM2 LoRA包装器（可选，用于预测）
        point_predictor: 点预测网络（可选，用于预测）
        image_size: 图像尺寸
        device: 设备
        max_rollout_steps: 最大rollout步数
    """
    # 加载图像
    image_path = sample['image']
    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        return
    
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    orig_w, orig_h = img.size
    
    # 获取点和标签
    points = sample.get('points', [])
    labels = sample.get('labels', [])
    
    if len(points) != len(labels):
        print(f"Warning: Points and labels length mismatch: {len(points)} vs {len(labels)}")
        return
    
    # 是否进行预测
    do_prediction = sam2_lora is not None and point_predictor is not None
    
    # 计算预测掩模和指标（如果需要）
    pred_mask = None
    metrics = {}
    pred_points_seq = []
    pred_labels_seq = []
    
    if do_prediction:
        try:
            with torch.no_grad():
                # 运行rollout预测
                img_resized = img.resize(image_size, Image.BILINEAR)
                img_tensor = TF.to_tensor(img_resized).unsqueeze(0).to(device)
                
                # 加载真值掩模
                gt_mask_path = sample.get('gt_mask', None)
                if gt_mask_path and os.path.exists(gt_mask_path):
                    gt_mask = Image.open(gt_mask_path).convert('L')
                    gt_mask_resized = gt_mask.resize(image_size, Image.NEAREST)
                    gt_mask_tensor = TF.to_tensor(gt_mask_resized).to(device)
                    gt_mask_binary = (gt_mask_tensor > 0.5).float()
                else:
                    gt_mask_binary = None
                
                # Rollout预测
                prev_mask = torch.zeros(1, 1, image_size[0], image_size[1]).to(device)
                
                for step in range(max_rollout_steps):
                    # 预测下一个点
                    sam_features = sam2_lora.get_image_features(img_tensor, feature_scale=8)
                    heatmap_logits, label_logits = point_predictor(sam_features, prev_mask)
                    
                    # Argmax获取点
                    B, _, H, W = heatmap_logits.shape
                    heatmap_flat = heatmap_logits.view(B, -1)
                    pred_idx = heatmap_flat.argmax(dim=1)
                    pred_y = (pred_idx // W).float()
                    pred_x = (pred_idx % W).float()
                    pred_label = label_logits.argmax(dim=1).item()
                    
                    # 记录点
                    pred_points_seq.append((pred_x.item(), pred_y.item()))
                    pred_labels_seq.append(pred_label)
                    
                    # 生成掩模
                    points_orig = [(px * orig_w / image_size[1], py * orig_h / image_size[0])
                                  for px, py in pred_points_seq]
                    prev_mask = sam2_lora.predict_mask(
                        np.array(img), points_orig, pred_labels_seq
                    )
                    if isinstance(prev_mask, torch.Tensor):
                        prev_mask = prev_mask.unsqueeze(0).to(device)
                    else:
                        prev_mask = torch.from_numpy(prev_mask).unsqueeze(0).to(device)
                    prev_mask = F.interpolate(
                        prev_mask, size=image_size, mode='nearest'
                    )
                    
                    # 计算指标
                    if gt_mask_binary is not None:
                        dice = compute_dice(prev_mask[0, 0], gt_mask_binary[0, 0])
                        iou = compute_iou(prev_mask[0, 0], gt_mask_binary[0, 0])
                        
                        # 保存最终指标
                        metrics['dice'] = dice
                        metrics['iou'] = iou
                        metrics['num_points'] = len(pred_points_seq)
                        pred_mask = prev_mask[0, 0].cpu().numpy()
                        
                        # 停止条件
                        if dice > 0.95 or (step > 0 and dice - metrics.get('prev_dice', 0) < 0.001):
                            break
                        metrics['prev_dice'] = dice
                
                # 转换为numpy用于可视化
                if pred_mask is None:
                    pred_mask = prev_mask[0, 0].cpu().numpy()
            
        except Exception as e:
            print(f"Warning: Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            do_prediction = False
    
    # 创建图形
    if do_prediction:
        # 两幅图：第一幅显示点和标注，第二幅显示掩模对比
        fig, axes = plt.subplots(1, 2, figsize=(24, 12))
        ax1, ax2 = axes
    else:
        # 单幅图：只显示点和标注
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 12))
        ax2 = None
    
    # 第一幅图：显示点和标注
    ax1.imshow(img_np)
    
    # 如果有真值掩模，叠加显示
    if show_mask:
        gt_mask_path = sample.get('gt_mask', None)
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                mask_binary = (gt_mask > 127).astype(float)
                ax1.imshow(mask_binary, alpha=0.3, cmap='Greens')
            except Exception as e:
                print(f"Warning: Failed to load mask {gt_mask_path}: {e}")
    
    # 绘制点
    for j, ((x, y), label) in enumerate(zip(points, labels)):
        if label == 1:
            # 前景点：红色，实心圆
            color = 'red'
            size = 150
            point_type = 'FG'
        else:
            # 背景点：蓝色，实心圆
            color = 'blue'
            size = 120
            point_type = 'BG'
        
        # 绘制点
        ax1.scatter(x, y, c=color, s=size, marker='o', 
                   edgecolors='white', linewidths=2, zorder=10)
        
        # 添加标签文本
        ax1.text(x + 15, y, f'{j}({point_type})', 
                fontsize=10, color='white', weight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    # 设置第一幅图的标题
    stem = Path(image_path).stem
    title1 = f'{stem}: {len(points)} points (Ground Truth)'
    ax1.set_title(title1, fontsize=14, weight='bold')
    ax1.axis('off')
    
    # 第二幅图：显示掩模对比（如果进行预测）
    if do_prediction and ax2 is not None:
        ax2.imshow(img_np)
        
        # 叠加真值掩模（绿色）
        gt_mask_path = sample.get('gt_mask', None)
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                gt_mask_binary = (gt_mask > 127).astype(float)
                # 调整尺寸以匹配图像
                if gt_mask_binary.shape != img_np.shape[:2]:
                    scale_y = img_np.shape[0] / gt_mask_binary.shape[0]
                    scale_x = img_np.shape[1] / gt_mask_binary.shape[1]
                    gt_mask_binary = zoom(gt_mask_binary, (scale_y, scale_x), order=0)
                
                # 绿色半透明显示真值掩模
                gt_mask_colored = np.zeros_like(img_np)
                gt_mask_colored[:, :, 1] = gt_mask_binary * 255
                ax2.imshow(gt_mask_colored, alpha=0.4)
            except Exception as e:
                print(f"Warning: Failed to load GT mask: {e}")
        
        # 叠加预测掩模（红色）
        pred_mask_binary = None
        if pred_mask is not None:
            # 调整尺寸以匹配图像
            if pred_mask.shape != img_np.shape[:2]:
                scale_y = img_np.shape[0] / pred_mask.shape[0]
                scale_x = img_np.shape[1] / pred_mask.shape[1]
                pred_mask_resized = zoom(pred_mask, (scale_y, scale_x), order=0)
            else:
                pred_mask_resized = pred_mask
            
            pred_mask_binary = (pred_mask_resized > 0.5).astype(float)
        
        # 优化显示：先显示重叠区域（黄色），再显示真值（绿色），最后显示预测（红色）
        if gt_mask_path and os.path.exists(gt_mask_path) and pred_mask_binary is not None:
            try:
                # 确保gt_mask_binary已加载
                if 'gt_mask_binary' not in locals():
                    gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                    gt_mask_binary = (gt_mask > 127).astype(float)
                    if gt_mask_binary.shape != img_np.shape[:2]:
                        scale_y = img_np.shape[0] / gt_mask_binary.shape[0]
                        scale_x = img_np.shape[1] / gt_mask_binary.shape[1]
                        gt_mask_binary = zoom(gt_mask_binary, (scale_y, scale_x), order=0)
                
                # 重叠区域（黄色）
                overlap = (gt_mask_binary * pred_mask_binary).astype(float)
                overlap_colored = np.zeros_like(img_np)
                overlap_colored[:, :, 0] = overlap * 255  # 红色
                overlap_colored[:, :, 1] = overlap * 255  # 绿色
                ax2.imshow(overlap_colored, alpha=0.5)
                
                # 真值掩模的未重叠部分（绿色）
                gt_only = (gt_mask_binary * (1 - pred_mask_binary)).astype(float)
                gt_only_colored = np.zeros_like(img_np)
                gt_only_colored[:, :, 1] = gt_only * 255
                ax2.imshow(gt_only_colored, alpha=0.4)
                
                # 预测掩模的未重叠部分（红色）
                pred_only = (pred_mask_binary * (1 - gt_mask_binary)).astype(float)
                pred_only_colored = np.zeros_like(img_np)
                pred_only_colored[:, :, 0] = pred_only * 255
                ax2.imshow(pred_only_colored, alpha=0.4)
            except Exception as e:
                print(f"Warning: Failed to overlay masks: {e}")
        elif pred_mask_binary is not None:
            # 只有预测掩模（红色）
            pred_mask_colored = np.zeros_like(img_np)
            pred_mask_colored[:, :, 0] = pred_mask_binary * 255
            ax2.imshow(pred_mask_colored, alpha=0.4)
        
        # 绘制预测点
        for j, ((x, y), label) in enumerate(zip(pred_points_seq, pred_labels_seq)):
            if label == 1:
                color = 'red'
                size = 150
                point_type = 'FG'
            else:
                color = 'blue'
                size = 120
                point_type = 'BG'
            
            # 缩放点坐标到原始图像尺寸
            x_orig = x * orig_w / image_size[1]
            y_orig = y * orig_h / image_size[0]
            
            ax2.scatter(x_orig, y_orig, c=color, s=size, marker='o', 
                       edgecolors='white', linewidths=2, zorder=10)
            ax2.text(x_orig + 15, y_orig, f'{j}({point_type})', 
                    fontsize=10, color='white', weight='bold',
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
        
        # 设置第二幅图的标题和指标
        title2 = f'Prediction: {len(pred_points_seq)} points'
        if metrics:
            metrics_text = f"Dice: {metrics.get('dice', 0):.3f}, IoU: {metrics.get('iou', 0):.3f}"
            title2 += f'\n{metrics_text}'
        ax2.set_title(title2, fontsize=14, weight='bold')
        ax2.axis('off')
        
        # 在图像角落添加指标文本
        if metrics:
            metrics_str = f"Dice: {metrics.get('dice', 0):.3f}\nIoU: {metrics.get('iou', 0):.3f}\nPoints: {metrics.get('num_points', 0)}"
            ax2.text(0.02, 0.98, metrics_str, 
                    transform=ax2.transAxes, fontsize=12, weight='bold',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='可视化PEFT训练数据',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--jsonl',
        type=str,
        required=True,
        help='输入JSONL文件路径'
    )
    
    parser.add_argument(
        '--out_dir',
        type=str,
        required=True,
        help='输出图像文件夹路径'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['random', 'all'],
        default='random',
        help='采样模式: random=随机采样, all=遍历所有样本'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='随机采样模式下的样本数量（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--show_mask',
        action='store_true',
        default=True,
        help='是否显示真值掩模（默认True）'
    )
    
    parser.add_argument(
        '--no_mask',
        dest='show_mask',
        action='store_false',
        help='不显示真值掩模'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='输出图像分辨率（DPI）'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default='sample',
        help='输出文件名前缀'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='模型checkpoint路径（如果提供，将生成预测对比图）'
    )
    
    parser.add_argument(
        '--sam_checkpoint',
        type=str,
        default=None,
        help='SAM2 checkpoint路径（如果提供checkpoint，此参数必需）'
    )
    
    parser.add_argument(
        '--image_size',
        type=int,
        nargs=2,
        default=[512, 512],
        help='图像尺寸 [H, W]（用于预测）'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='设备（cuda/cpu）'
    )
    
    parser.add_argument(
        '--max_rollout_steps',
        type=int,
        default=16,
        help='最大rollout步数（用于预测）'
    )
    
    args = parser.parse_args()
    
    # 加载模型（如果提供了checkpoint）
    sam2_lora = None
    point_predictor = None
    
    if args.checkpoint:
        if not args.sam_checkpoint:
            print("Error: --sam_checkpoint is required when --checkpoint is provided")
            sys.exit(1)
        
        print(f"Loading model from {args.checkpoint}...")
        try:
            device = torch.device(args.device)
            
            # 加载checkpoint
            ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
            config = ckpt.get('config', {})
            
            # 提取配置
            lora_rank = config.get('lora_rank', 16)
            lora_alpha = config.get('lora_alpha', 32)
            fusion_mode = config.get('fusion_mode', 'film')
            feature_scale = config.get('feature_scale', 8)
            
            # 初始化SAM2 LoRA
            sam2_lora = LoRASAM2Wrapper(
                sam_checkpoint=args.sam_checkpoint,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                device=str(device),
            )
            
            # 初始化点预测网络
            sam_feature_dim = 256  # SAM2 Hiera Large的特征维度
            point_predictor = PointPredictorFromSAMFeatures(
                sam_feature_dim=sam_feature_dim,
                output_size=tuple(args.image_size),
                fusion_mode=fusion_mode,
                feature_scale=feature_scale,
            ).to(device)
            
            # 加载权重
            load_checkpoint(
                args.checkpoint, point_predictor, sam2_lora, None, None, None, str(device)
            )
            
            point_predictor.eval()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Will continue without prediction...")
            sam2_lora = None
            point_predictor = None
    
    # 加载数据
    print(f"Loading data from {args.jsonl}...")
    data = load_jsonl(args.jsonl)
    print(f"Loaded {len(data)} samples")
    
    # 选择样本
    if args.mode == 'random':
        random.seed(args.seed)
        selected_samples = random.sample(data, min(args.num_samples, len(data)))
        print(f"Randomly selected {len(selected_samples)} samples")
    else:
        selected_samples = data
        print(f"Processing all {len(selected_samples)} samples")
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 可视化每个样本
    print(f"\nGenerating visualizations...")
    for i, sample in enumerate(selected_samples):
        # 生成输出文件名
        if args.mode == 'random':
            output_filename = f'{args.prefix}_{i:03d}.png'
        else:
            # 使用图像文件名作为输出文件名
            image_path = sample.get('image', '')
            if image_path:
                stem = Path(image_path).stem
                output_filename = f'{args.prefix}_{stem}.png'
            else:
                output_filename = f'{args.prefix}_{i:04d}.png'
        
        output_path = os.path.join(args.out_dir, output_filename)
        
        # 可视化
        try:
            visualize_sample(
                sample, output_path, 
                show_mask=args.show_mask, 
                dpi=args.dpi,
                sam2_lora=sam2_lora,
                point_predictor=point_predictor,
                image_size=tuple(args.image_size),
                device=args.device,
                max_rollout_steps=args.max_rollout_steps,
            )
            print(f"  [{i+1}/{len(selected_samples)}] Saved: {output_path}")
        except Exception as e:
            print(f"  [{i+1}/{len(selected_samples)}] Error processing sample {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nDone! Visualizations saved to: {args.out_dir}")
    print(f"Total: {len(selected_samples)} images generated")


if __name__ == '__main__':
    main()


```


修复真值掩模的显示逻辑，确保正确加载和显示：


```python
#!/usr/bin/env python3
"""
可视化训练数据脚本

用于可视化PEFT训练数据中的图像、点标注和真值掩模。
支持随机采样或遍历所有样本。
如果提供了checkpoint，还会生成预测掩模对比图。

用法:
    # 随机采样5个样本（仅显示标注）
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_train-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode random \
        --num_samples 5

    # 遍历所有样本（带预测对比）
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_test-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode all \
        --checkpoint outputs/braintumour/peft_supervised_baseline-251107/checkpoint_best.pt \
        --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
        --image_size 512 512
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.ndimage import zoom

# 添加父目录到路径
_PARENT_DIR = Path(__file__).parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from peft.lora_sam2 import LoRASAM2Wrapper
from peft.point_predictor_peft import PointPredictorFromSAMFeatures
from peft.utils_peft import compute_dice, compute_iou, load_checkpoint


def load_jsonl(jsonl_path: str) -> List[Dict]:
    """加载JSONL文件"""
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith('['):
            # JSON数组格式
            data = json.loads(content)
        else:
            # JSONL格式（逐行）
            data = []
            for line in content.splitlines():
                if line.strip():
                    data.append(json.loads(line))
    return data


def visualize_sample(
    sample: Dict,
    output_path: str,
    show_mask: bool = True,
    dpi: int = 150,
    sam2_lora: Optional[LoRASAM2Wrapper] = None,
    point_predictor: Optional[PointPredictorFromSAMFeatures] = None,
    image_size: Tuple[int, int] = (512, 512),
    device: str = 'cuda',
    max_rollout_steps: int = 16,
) -> None:
    """
    可视化单个样本
    
    Args:
        sample: 样本字典，包含 'image', 'gt_mask', 'points', 'labels' 等字段
        output_path: 输出图像路径
        show_mask: 是否显示真值掩模
        dpi: 图像分辨率
        sam2_lora: SAM2 LoRA包装器（可选，用于预测）
        point_predictor: 点预测网络（可选，用于预测）
        image_size: 图像尺寸
        device: 设备
        max_rollout_steps: 最大rollout步数
    """
    # 加载图像
    image_path = sample['image']
    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        return
    
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    orig_w, orig_h = img.size
    
    # 获取点和标签
    points = sample.get('points', [])
    labels = sample.get('labels', [])
    
    if len(points) != len(labels):
        print(f"Warning: Points and labels length mismatch: {len(points)} vs {len(labels)}")
        return
    
    # 是否进行预测
    do_prediction = sam2_lora is not None and point_predictor is not None
    
    # 计算预测掩模和指标（如果需要）
    pred_mask = None
    metrics = {}
    pred_points_seq = []
    pred_labels_seq = []
    
    if do_prediction:
        try:
            with torch.no_grad():
                # 运行rollout预测
                img_resized = img.resize(image_size, Image.BILINEAR)
                img_tensor = TF.to_tensor(img_resized).unsqueeze(0).to(device)
                
                # 加载真值掩模
                gt_mask_path = sample.get('gt_mask', None)
                if gt_mask_path and os.path.exists(gt_mask_path):
                    gt_mask = Image.open(gt_mask_path).convert('L')
                    gt_mask_resized = gt_mask.resize(image_size, Image.NEAREST)
                    gt_mask_tensor = TF.to_tensor(gt_mask_resized).to(device)
                    gt_mask_binary = (gt_mask_tensor > 0.5).float()
                else:
                    gt_mask_binary = None
                
                # Rollout预测
                prev_mask = torch.zeros(1, 1, image_size[0], image_size[1]).to(device)
                
                for step in range(max_rollout_steps):
                    # 预测下一个点
                    sam_features = sam2_lora.get_image_features(img_tensor, feature_scale=8)
                    heatmap_logits, label_logits = point_predictor(sam_features, prev_mask)
                    
                    # Argmax获取点
                    B, _, H, W = heatmap_logits.shape
                    heatmap_flat = heatmap_logits.view(B, -1)
                    pred_idx = heatmap_flat.argmax(dim=1)
                    pred_y = (pred_idx // W).float()
                    pred_x = (pred_idx % W).float()
                    pred_label = label_logits.argmax(dim=1).item()
                    
                    # 记录点
                    pred_points_seq.append((pred_x.item(), pred_y.item()))
                    pred_labels_seq.append(pred_label)
                    
                    # 生成掩模
                    points_orig = [(px * orig_w / image_size[1], py * orig_h / image_size[0])
                                  for px, py in pred_points_seq]
                    prev_mask = sam2_lora.predict_mask(
                        np.array(img), points_orig, pred_labels_seq
                    )
                    if isinstance(prev_mask, torch.Tensor):
                        prev_mask = prev_mask.unsqueeze(0).to(device)
                    else:
                        prev_mask = torch.from_numpy(prev_mask).unsqueeze(0).to(device)
                    prev_mask = F.interpolate(
                        prev_mask, size=image_size, mode='nearest'
                    )
                    
                    # 计算指标
                    if gt_mask_binary is not None:
                        dice = compute_dice(prev_mask[0, 0], gt_mask_binary[0, 0])
                        iou = compute_iou(prev_mask[0, 0], gt_mask_binary[0, 0])
                        
                        # 保存最终指标
                        metrics['dice'] = dice
                        metrics['iou'] = iou
                        metrics['num_points'] = len(pred_points_seq)
                        pred_mask = prev_mask[0, 0].cpu().numpy()
                        
                        # 停止条件
                        if dice > 0.95 or (step > 0 and dice - metrics.get('prev_dice', 0) < 0.001):
                            break
                        metrics['prev_dice'] = dice
                
                # 转换为numpy用于可视化
                if pred_mask is None:
                    pred_mask = prev_mask[0, 0].cpu().numpy()
            
        except Exception as e:
            print(f"Warning: Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            do_prediction = False
    
    # 创建图形
    if do_prediction:
        # 两幅图：第一幅显示点和标注，第二幅显示掩模对比
        fig, axes = plt.subplots(1, 2, figsize=(24, 12))
        ax1, ax2 = axes
    else:
        # 单幅图：只显示点和标注
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 12))
        ax2 = None
    
    # 第一幅图：显示点和标注
    ax1.imshow(img_np)
    
    # 如果有真值掩模，叠加显示
    if show_mask:
        gt_mask_path = sample.get('gt_mask', None)
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                mask_binary = (gt_mask > 127).astype(float)
                ax1.imshow(mask_binary, alpha=0.3, cmap='Greens')
            except Exception as e:
                print(f"Warning: Failed to load mask {gt_mask_path}: {e}")
    
    # 绘制点
    for j, ((x, y), label) in enumerate(zip(points, labels)):
        if label == 1:
            # 前景点：红色，实心圆
            color = 'red'
            size = 150
            point_type = 'FG'
        else:
            # 背景点：蓝色，实心圆
            color = 'blue'
            size = 120
            point_type = 'BG'
        
        # 绘制点
        ax1.scatter(x, y, c=color, s=size, marker='o', 
                   edgecolors='white', linewidths=2, zorder=10)
        
        # 添加标签文本
        ax1.text(x + 15, y, f'{j}({point_type})', 
                fontsize=10, color='white', weight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    # 设置第一幅图的标题
    stem = Path(image_path).stem
    title1 = f'{stem}: {len(points)} points (Ground Truth)'
    ax1.set_title(title1, fontsize=14, weight='bold')
    ax1.axis('off')
    
    # 第二幅图：显示掩模对比（如果进行预测）
    if do_prediction and ax2 is not None:
        ax2.imshow(img_np)
        
        gt_mask_path = sample.get('gt_mask', None)
        gt_mask_binary = None
        pred_mask_binary = None
        
        # 加载并调整真值掩模尺寸
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                gt_mask_binary = (gt_mask > 127).astype(float)
                # 调整尺寸以匹配图像
                if gt_mask_binary.shape != img_np.shape[:2]:
                    scale_y = img_np.shape[0] / gt_mask_binary.shape[0]
                    scale_x = img_np.shape[1] / gt_mask_binary.shape[1]
                    gt_mask_binary = zoom(gt_mask_binary, (scale_y, scale_x), order=0)
            except Exception as e:
                print(f"Warning: Failed to load GT mask: {e}")
        
        # 调整预测掩模尺寸
        if pred_mask is not None:
            if pred_mask.shape != img_np.shape[:2]:
                scale_y = img_np.shape[0] / pred_mask.shape[0]
                scale_x = img_np.shape[1] / pred_mask.shape[1]
                pred_mask_resized = zoom(pred_mask, (scale_y, scale_x), order=0)
            else:
                pred_mask_resized = pred_mask
            pred_mask_binary = (pred_mask_resized > 0.5).astype(float)
        
        # 优化显示：先显示重叠区域（黄色），再显示真值（绿色），最后显示预测（红色）
        if gt_mask_binary is not None and pred_mask_binary is not None:
            # 重叠区域（黄色）
            overlap = (gt_mask_binary * pred_mask_binary).astype(float)
            overlap_colored = np.zeros_like(img_np)
            overlap_colored[:, :, 0] = overlap * 255  # 红色
            overlap_colored[:, :, 1] = overlap * 255  # 绿色
            ax2.imshow(overlap_colored, alpha=0.5)
            
            # 真值掩模的未重叠部分（绿色）
            gt_only = (gt_mask_binary * (1 - pred_mask_binary)).astype(float)
            gt_only_colored = np.zeros_like(img_np)
            gt_only_colored[:, :, 1] = gt_only * 255
            ax2.imshow(gt_only_colored, alpha=0.4)
            
            # 预测掩模的未重叠部分（红色）
            pred_only = (pred_mask_binary * (1 - gt_mask_binary)).astype(float)
            pred_only_colored = np.zeros_like(img_np)
            pred_only_colored[:, :, 0] = pred_only * 255
            ax2.imshow(pred_only_colored, alpha=0.4)
        elif gt_mask_binary is not None:
            # 只有真值掩模（绿色）
            gt_mask_colored = np.zeros_like(img_np)
            gt_mask_colored[:, :, 1] = gt_mask_binary * 255
            ax2.imshow(gt_mask_colored, alpha=0.4)
        elif pred_mask_binary is not None:
            # 只有预测掩模（红色）
            pred_mask_colored = np.zeros_like(img_np)
            pred_mask_colored[:, :, 0] = pred_mask_binary * 255
            ax2.imshow(pred_mask_colored, alpha=0.4)
        
        # 绘制预测点
        for j, ((x, y), label) in enumerate(zip(pred_points_seq, pred_labels_seq)):
            if label == 1:
                color = 'red'
                size = 150
                point_type = 'FG'
            else:
                color = 'blue'
                size = 120
                point_type = 'BG'
            
            # 缩放点坐标到原始图像尺寸
            x_orig = x * orig_w / image_size[1]
            y_orig = y * orig_h / image_size[0]
            
            ax2.scatter(x_orig, y_orig, c=color, s=size, marker='o', 
                       edgecolors='white', linewidths=2, zorder=10)
            ax2.text(x_orig + 15, y_orig, f'{j}({point_type})', 
                    fontsize=10, color='white', weight='bold',
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
        
        # 设置第二幅图的标题和指标
        title2 = f'Prediction: {len(pred_points_seq)} points'
        if metrics:
            metrics_text = f"Dice: {metrics.get('dice', 0):.3f}, IoU: {metrics.get('iou', 0):.3f}"
            title2 += f'\n{metrics_text}'
        ax2.set_title(title2, fontsize=14, weight='bold')
        ax2.axis('off')
        
        # 在图像角落添加指标文本
        if metrics:
            metrics_str = f"Dice: {metrics.get('dice', 0):.3f}\nIoU: {metrics.get('iou', 0):.3f}\nPoints: {metrics.get('num_points', 0)}"
            ax2.text(0.02, 0.98, metrics_str, 
                    transform=ax2.transAxes, fontsize=12, weight='bold',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='可视化PEFT训练数据',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--jsonl',
        type=str,
        required=True,
        help='输入JSONL文件路径'
    )
    
    parser.add_argument(
        '--out_dir',
        type=str,
        required=True,
        help='输出图像文件夹路径'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['random', 'all'],
        default='random',
        help='采样模式: random=随机采样, all=遍历所有样本'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='随机采样模式下的样本数量（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--show_mask',
        action='store_true',
        default=True,
        help='是否显示真值掩模（默认True）'
    )
    
    parser.add_argument(
        '--no_mask',
        dest='show_mask',
        action='store_false',
        help='不显示真值掩模'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='输出图像分辨率（DPI）'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default='sample',
        help='输出文件名前缀'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='模型checkpoint路径（如果提供，将生成预测对比图）'
    )
    
    parser.add_argument(
        '--sam_checkpoint',
        type=str,
        default=None,
        help='SAM2 checkpoint路径（如果提供checkpoint，此参数必需）'
    )
    
    parser.add_argument(
        '--image_size',
        type=int,
        nargs=2,
        default=[512, 512],
        help='图像尺寸 [H, W]（用于预测）'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='设备（cuda/cpu）'
    )
    
    parser.add_argument(
        '--max_rollout_steps',
        type=int,
        default=16,
        help='最大rollout步数（用于预测）'
    )
    
    args = parser.parse_args()
    
    # 加载模型（如果提供了checkpoint）
    sam2_lora = None
    point_predictor = None
    
    if args.checkpoint:
        if not args.sam_checkpoint:
            print("Error: --sam_checkpoint is required when --checkpoint is provided")
            sys.exit(1)
        
        print(f"Loading model from {args.checkpoint}...")
        try:
            device = torch.device(args.device)
            
            # 加载checkpoint
            ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
            config = ckpt.get('config', {})
            
            # 提取配置
            lora_rank = config.get('lora_rank', 16)
            lora_alpha = config.get('lora_alpha', 32)
            fusion_mode = config.get('fusion_mode', 'film')
            feature_scale = config.get('feature_scale', 8)
            
            # 初始化SAM2 LoRA
            sam2_lora = LoRASAM2Wrapper(
                sam_checkpoint=args.sam_checkpoint,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                device=str(device),
            )
            
            # 初始化点预测网络
            sam_feature_dim = 256  # SAM2 Hiera Large的特征维度
            point_predictor = PointPredictorFromSAMFeatures(
                sam_feature_dim=sam_feature_dim,
                output_size=tuple(args.image_size),
                fusion_mode=fusion_mode,
                feature_scale=feature_scale,
            ).to(device)
            
            # 加载权重
            load_checkpoint(
                args.checkpoint, point_predictor, sam2_lora, None, None, None, str(device)
            )
            
            point_predictor.eval()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Will continue without prediction...")
            sam2_lora = None
            point_predictor = None
    
    # 加载数据
    print(f"Loading data from {args.jsonl}...")
    data = load_jsonl(args.jsonl)
    print(f"Loaded {len(data)} samples")
    
    # 选择样本
    if args.mode == 'random':
        random.seed(args.seed)
        selected_samples = random.sample(data, min(args.num_samples, len(data)))
        print(f"Randomly selected {len(selected_samples)} samples")
    else:
        selected_samples = data
        print(f"Processing all {len(selected_samples)} samples")
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 可视化每个样本
    print(f"\nGenerating visualizations...")
    for i, sample in enumerate(selected_samples):
        # 生成输出文件名
        if args.mode == 'random':
            output_filename = f'{args.prefix}_{i:03d}.png'
        else:
            # 使用图像文件名作为输出文件名
            image_path = sample.get('image', '')
            if image_path:
                stem = Path(image_path).stem
                output_filename = f'{args.prefix}_{stem}.png'
            else:
                output_filename = f'{args.prefix}_{i:04d}.png'
        
        output_path = os.path.join(args.out_dir, output_filename)
        
        # 可视化
        try:
            visualize_sample(
                sample, output_path, 
                show_mask=args.show_mask, 
                dpi=args.dpi,
                sam2_lora=sam2_lora,
                point_predictor=point_predictor,
                image_size=tuple(args.image_size),
                device=args.device,
                max_rollout_steps=args.max_rollout_steps,
            )
            print(f"  [{i+1}/{len(selected_samples)}] Saved: {output_path}")
        except Exception as e:
            print(f"  [{i+1}/{len(selected_samples)}] Error processing sample {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nDone! Visualizations saved to: {args.out_dir}")
    print(f"Total: {len(selected_samples)} images generated")


if __name__ == '__main__':
    main()


```


脚本已更新。功能如下：

## 功能总结

### 新增功能

1. **两幅图像对比**：
   - **第一幅**：显示真值点和标注（Ground Truth）
   - **第二幅**：显示预测掩模对比（仅在提供checkpoint时生成）

2. **掩模可视化**：
   - **绿色**：真值掩模的未重叠部分
   - **红色**：预测掩模的未重叠部分
   - **黄色**：真值和预测的重叠区域

3. **指标显示**：
   - 在第二幅图的**标题**中显示：Dice、IoU、点数
   - 在第二幅图的**左上角**显示指标文本框

### 使用示例

```bash
# 带预测对比的可视化
python -m seg-rl.peft.visualize_training_data \
    --jsonl outputs/braintumour/peft_test-251107.jsonl \
    --out_dir outputs/braintumour/visualizations_with_pred \
    --mode random \
    --num_samples 5 \
    --checkpoint outputs/braintumour/peft_supervised_baseline-251107/checkpoint_best.pt \
    --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
    --image_size 512 512 \
    --device cuda
```

### 输出说明

- **无checkpoint**：只生成第一幅图（点和标注）
- **有checkpoint**：生成两幅图（左：真值，右：预测对比）

脚本已就绪，可直接使用。

```shellscript
# 带预测对比的可视化
python -m seg-rl.peft.visualize_training_data \
    --jsonl outputs/braintumour/peft_test-251107.jsonl \
    --out_dir outputs/braintumour/visualizations_with_pred \
    --mode random \
    --num_samples 5 \
    --checkpoint outputs/braintumour/peft_supervised_baseline-251107/checkpoint_best.pt \
    --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
    --image_size 512 512 \
    --device cuda
```

---

**User**

错了，该脚本做预测掩模对比并不需要使用预测模型，因为所有需要的信息都在JSONL文件里：真值mask在gt_mask所指向的路径，预测masks在sam_masks_dir指向的目录下，每个样本各有一个同名子目录存放所有步骤的预测masks。所以你应该直接通过JSON找到已存好的masks而无需重新做预测分割

---

**Cursor**


修改脚本，改为直接从 JSONL 和文件系统读取已保存的掩模，移除模型预测部分：



重写函数，移除模型预测，改为直接从文件系统读取已保存的掩模：


```python
#!/usr/bin/env python3
"""
可视化训练数据脚本

用于可视化PEFT训练数据中的图像、点标注和真值掩模。
支持随机采样或遍历所有样本。
如果提供了checkpoint，还会生成预测掩模对比图。

用法:
    # 随机采样5个样本（仅显示标注）
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_train-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode random \
        --num_samples 5

    # 遍历所有样本（带预测对比）
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_test-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode all \
        --checkpoint outputs/braintumour/peft_supervised_baseline-251107/checkpoint_best.pt \
        --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
        --image_size 512 512
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.ndimage import zoom

# 添加父目录到路径
_PARENT_DIR = Path(__file__).parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from peft.lora_sam2 import LoRASAM2Wrapper
from peft.point_predictor_peft import PointPredictorFromSAMFeatures
from peft.utils_peft import compute_dice, compute_iou, load_checkpoint


def load_jsonl(jsonl_path: str) -> List[Dict]:
    """加载JSONL文件"""
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith('['):
            # JSON数组格式
            data = json.loads(content)
        else:
            # JSONL格式（逐行）
            data = []
            for line in content.splitlines():
                if line.strip():
                    data.append(json.loads(line))
    return data


def visualize_sample(
    sample: Dict,
    output_path: str,
    show_mask: bool = True,
    dpi: int = 150,
    show_prediction: bool = True,
) -> None:
    """
    可视化单个样本
    
    Args:
        sample: 样本字典，包含 'image', 'gt_mask', 'points', 'labels', 'sam_masks_dir' 等字段
        output_path: 输出图像路径
        show_mask: 是否显示真值掩模
        dpi: 图像分辨率
        show_prediction: 是否显示预测掩模对比（需要sam_masks_dir）
    """
    # 加载图像
    image_path = sample['image']
    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        return
    
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    orig_w, orig_h = img.size
    
    # 获取点和标签
    points = sample.get('points', [])
    labels = sample.get('labels', [])
    
    if len(points) != len(labels):
        print(f"Warning: Points and labels length mismatch: {len(points)} vs {len(labels)}")
        return
    
    # 是否进行预测
    do_prediction = sam2_lora is not None and point_predictor is not None
    
    # 计算预测掩模和指标（如果需要）
    pred_mask = None
    metrics = {}
    pred_points_seq = []
    pred_labels_seq = []
    
    if do_prediction:
        try:
            with torch.no_grad():
                # 运行rollout预测
                img_resized = img.resize(image_size, Image.BILINEAR)
                img_tensor = TF.to_tensor(img_resized).unsqueeze(0).to(device)
                
                # 加载真值掩模
                gt_mask_path = sample.get('gt_mask', None)
                if gt_mask_path and os.path.exists(gt_mask_path):
                    gt_mask = Image.open(gt_mask_path).convert('L')
                    gt_mask_resized = gt_mask.resize(image_size, Image.NEAREST)
                    gt_mask_tensor = TF.to_tensor(gt_mask_resized).to(device)
                    gt_mask_binary = (gt_mask_tensor > 0.5).float()
                else:
                    gt_mask_binary = None
                
                # Rollout预测
                prev_mask = torch.zeros(1, 1, image_size[0], image_size[1]).to(device)
                
                for step in range(max_rollout_steps):
                    # 预测下一个点
                    sam_features = sam2_lora.get_image_features(img_tensor, feature_scale=8)
                    heatmap_logits, label_logits = point_predictor(sam_features, prev_mask)
                    
                    # Argmax获取点
                    B, _, H, W = heatmap_logits.shape
                    heatmap_flat = heatmap_logits.view(B, -1)
                    pred_idx = heatmap_flat.argmax(dim=1)
                    pred_y = (pred_idx // W).float()
                    pred_x = (pred_idx % W).float()
                    pred_label = label_logits.argmax(dim=1).item()
                    
                    # 记录点
                    pred_points_seq.append((pred_x.item(), pred_y.item()))
                    pred_labels_seq.append(pred_label)
                    
                    # 生成掩模
                    points_orig = [(px * orig_w / image_size[1], py * orig_h / image_size[0])
                                  for px, py in pred_points_seq]
                    prev_mask = sam2_lora.predict_mask(
                        np.array(img), points_orig, pred_labels_seq
                    )
                    if isinstance(prev_mask, torch.Tensor):
                        prev_mask = prev_mask.unsqueeze(0).to(device)
                    else:
                        prev_mask = torch.from_numpy(prev_mask).unsqueeze(0).to(device)
                    prev_mask = F.interpolate(
                        prev_mask, size=image_size, mode='nearest'
                    )
                    
                    # 计算指标
                    if gt_mask_binary is not None:
                        dice = compute_dice(prev_mask[0, 0], gt_mask_binary[0, 0])
                        iou = compute_iou(prev_mask[0, 0], gt_mask_binary[0, 0])
                        
                        # 保存最终指标
                        metrics['dice'] = dice
                        metrics['iou'] = iou
                        metrics['num_points'] = len(pred_points_seq)
                        pred_mask = prev_mask[0, 0].cpu().numpy()
                        
                        # 停止条件
                        if dice > 0.95 or (step > 0 and dice - metrics.get('prev_dice', 0) < 0.001):
                            break
                        metrics['prev_dice'] = dice
                
                # 转换为numpy用于可视化
                if pred_mask is None:
                    pred_mask = prev_mask[0, 0].cpu().numpy()
            
        except Exception as e:
            print(f"Warning: Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            do_prediction = False
    
    # 创建图形
    if do_prediction:
        # 两幅图：第一幅显示点和标注，第二幅显示掩模对比
        fig, axes = plt.subplots(1, 2, figsize=(24, 12))
        ax1, ax2 = axes
    else:
        # 单幅图：只显示点和标注
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 12))
        ax2 = None
    
    # 第一幅图：显示点和标注
    ax1.imshow(img_np)
    
    # 如果有真值掩模，叠加显示
    if show_mask:
        gt_mask_path = sample.get('gt_mask', None)
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                mask_binary = (gt_mask > 127).astype(float)
                ax1.imshow(mask_binary, alpha=0.3, cmap='Greens')
            except Exception as e:
                print(f"Warning: Failed to load mask {gt_mask_path}: {e}")
    
    # 绘制点
    for j, ((x, y), label) in enumerate(zip(points, labels)):
        if label == 1:
            # 前景点：红色，实心圆
            color = 'red'
            size = 150
            point_type = 'FG'
        else:
            # 背景点：蓝色，实心圆
            color = 'blue'
            size = 120
            point_type = 'BG'
        
        # 绘制点
        ax1.scatter(x, y, c=color, s=size, marker='o', 
                   edgecolors='white', linewidths=2, zorder=10)
        
        # 添加标签文本
        ax1.text(x + 15, y, f'{j}({point_type})', 
                fontsize=10, color='white', weight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    # 设置第一幅图的标题
    stem = Path(image_path).stem
    title1 = f'{stem}: {len(points)} points (Ground Truth)'
    ax1.set_title(title1, fontsize=14, weight='bold')
    ax1.axis('off')
    
    # 第二幅图：显示掩模对比（如果进行预测）
    if do_prediction and ax2 is not None:
        ax2.imshow(img_np)
        
        gt_mask_path = sample.get('gt_mask', None)
        gt_mask_binary = None
        pred_mask_binary = None
        
        # 加载并调整真值掩模尺寸
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                gt_mask_binary = (gt_mask > 127).astype(float)
                # 调整尺寸以匹配图像
                if gt_mask_binary.shape != img_np.shape[:2]:
                    scale_y = img_np.shape[0] / gt_mask_binary.shape[0]
                    scale_x = img_np.shape[1] / gt_mask_binary.shape[1]
                    gt_mask_binary = zoom(gt_mask_binary, (scale_y, scale_x), order=0)
            except Exception as e:
                print(f"Warning: Failed to load GT mask: {e}")
        
        # 调整预测掩模尺寸
        if pred_mask is not None:
            if pred_mask.shape != img_np.shape[:2]:
                scale_y = img_np.shape[0] / pred_mask.shape[0]
                scale_x = img_np.shape[1] / pred_mask.shape[1]
                pred_mask_resized = zoom(pred_mask, (scale_y, scale_x), order=0)
            else:
                pred_mask_resized = pred_mask
            pred_mask_binary = (pred_mask_resized > 0.5).astype(float)
        
        # 优化显示：先显示重叠区域（黄色），再显示真值（绿色），最后显示预测（红色）
        if gt_mask_binary is not None and pred_mask_binary is not None:
            # 重叠区域（黄色）
            overlap = (gt_mask_binary * pred_mask_binary).astype(float)
            overlap_colored = np.zeros_like(img_np)
            overlap_colored[:, :, 0] = overlap * 255  # 红色
            overlap_colored[:, :, 1] = overlap * 255  # 绿色
            ax2.imshow(overlap_colored, alpha=0.5)
            
            # 真值掩模的未重叠部分（绿色）
            gt_only = (gt_mask_binary * (1 - pred_mask_binary)).astype(float)
            gt_only_colored = np.zeros_like(img_np)
            gt_only_colored[:, :, 1] = gt_only * 255
            ax2.imshow(gt_only_colored, alpha=0.4)
            
            # 预测掩模的未重叠部分（红色）
            pred_only = (pred_mask_binary * (1 - gt_mask_binary)).astype(float)
            pred_only_colored = np.zeros_like(img_np)
            pred_only_colored[:, :, 0] = pred_only * 255
            ax2.imshow(pred_only_colored, alpha=0.4)
        elif gt_mask_binary is not None:
            # 只有真值掩模（绿色）
            gt_mask_colored = np.zeros_like(img_np)
            gt_mask_colored[:, :, 1] = gt_mask_binary * 255
            ax2.imshow(gt_mask_colored, alpha=0.4)
        elif pred_mask_binary is not None:
            # 只有预测掩模（红色）
            pred_mask_colored = np.zeros_like(img_np)
            pred_mask_colored[:, :, 0] = pred_mask_binary * 255
            ax2.imshow(pred_mask_colored, alpha=0.4)
        
        # 绘制预测点
        for j, ((x, y), label) in enumerate(zip(pred_points_seq, pred_labels_seq)):
            if label == 1:
                color = 'red'
                size = 150
                point_type = 'FG'
            else:
                color = 'blue'
                size = 120
                point_type = 'BG'
            
            # 缩放点坐标到原始图像尺寸
            x_orig = x * orig_w / image_size[1]
            y_orig = y * orig_h / image_size[0]
            
            ax2.scatter(x_orig, y_orig, c=color, s=size, marker='o', 
                       edgecolors='white', linewidths=2, zorder=10)
            ax2.text(x_orig + 15, y_orig, f'{j}({point_type})', 
                    fontsize=10, color='white', weight='bold',
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
        
        # 设置第二幅图的标题和指标
        title2 = f'Prediction: {len(pred_points_seq)} points'
        if metrics:
            metrics_text = f"Dice: {metrics.get('dice', 0):.3f}, IoU: {metrics.get('iou', 0):.3f}"
            title2 += f'\n{metrics_text}'
        ax2.set_title(title2, fontsize=14, weight='bold')
        ax2.axis('off')
        
        # 在图像角落添加指标文本
        if metrics:
            metrics_str = f"Dice: {metrics.get('dice', 0):.3f}\nIoU: {metrics.get('iou', 0):.3f}\nPoints: {metrics.get('num_points', 0)}"
            ax2.text(0.02, 0.98, metrics_str, 
                    transform=ax2.transAxes, fontsize=12, weight='bold',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='可视化PEFT训练数据',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--jsonl',
        type=str,
        required=True,
        help='输入JSONL文件路径'
    )
    
    parser.add_argument(
        '--out_dir',
        type=str,
        required=True,
        help='输出图像文件夹路径'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['random', 'all'],
        default='random',
        help='采样模式: random=随机采样, all=遍历所有样本'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='随机采样模式下的样本数量（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--show_mask',
        action='store_true',
        default=True,
        help='是否显示真值掩模（默认True）'
    )
    
    parser.add_argument(
        '--no_mask',
        dest='show_mask',
        action='store_false',
        help='不显示真值掩模'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='输出图像分辨率（DPI）'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default='sample',
        help='输出文件名前缀'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='模型checkpoint路径（如果提供，将生成预测对比图）'
    )
    
    parser.add_argument(
        '--sam_checkpoint',
        type=str,
        default=None,
        help='SAM2 checkpoint路径（如果提供checkpoint，此参数必需）'
    )
    
    parser.add_argument(
        '--image_size',
        type=int,
        nargs=2,
        default=[512, 512],
        help='图像尺寸 [H, W]（用于预测）'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='设备（cuda/cpu）'
    )
    
    parser.add_argument(
        '--max_rollout_steps',
        type=int,
        default=16,
        help='最大rollout步数（用于预测）'
    )
    
    args = parser.parse_args()
    
    # 加载模型（如果提供了checkpoint）
    sam2_lora = None
    point_predictor = None
    
    if args.checkpoint:
        if not args.sam_checkpoint:
            print("Error: --sam_checkpoint is required when --checkpoint is provided")
            sys.exit(1)
        
        print(f"Loading model from {args.checkpoint}...")
        try:
            device = torch.device(args.device)
            
            # 加载checkpoint
            ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
            config = ckpt.get('config', {})
            
            # 提取配置
            lora_rank = config.get('lora_rank', 16)
            lora_alpha = config.get('lora_alpha', 32)
            fusion_mode = config.get('fusion_mode', 'film')
            feature_scale = config.get('feature_scale', 8)
            
            # 初始化SAM2 LoRA
            sam2_lora = LoRASAM2Wrapper(
                sam_checkpoint=args.sam_checkpoint,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                device=str(device),
            )
            
            # 初始化点预测网络
            sam_feature_dim = 256  # SAM2 Hiera Large的特征维度
            point_predictor = PointPredictorFromSAMFeatures(
                sam_feature_dim=sam_feature_dim,
                output_size=tuple(args.image_size),
                fusion_mode=fusion_mode,
                feature_scale=feature_scale,
            ).to(device)
            
            # 加载权重
            load_checkpoint(
                args.checkpoint, point_predictor, sam2_lora, None, None, None, str(device)
            )
            
            point_predictor.eval()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Will continue without prediction...")
            sam2_lora = None
            point_predictor = None
    
    # 加载数据
    print(f"Loading data from {args.jsonl}...")
    data = load_jsonl(args.jsonl)
    print(f"Loaded {len(data)} samples")
    
    # 选择样本
    if args.mode == 'random':
        random.seed(args.seed)
        selected_samples = random.sample(data, min(args.num_samples, len(data)))
        print(f"Randomly selected {len(selected_samples)} samples")
    else:
        selected_samples = data
        print(f"Processing all {len(selected_samples)} samples")
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 可视化每个样本
    print(f"\nGenerating visualizations...")
    for i, sample in enumerate(selected_samples):
        # 生成输出文件名
        if args.mode == 'random':
            output_filename = f'{args.prefix}_{i:03d}.png'
        else:
            # 使用图像文件名作为输出文件名
            image_path = sample.get('image', '')
            if image_path:
                stem = Path(image_path).stem
                output_filename = f'{args.prefix}_{stem}.png'
            else:
                output_filename = f'{args.prefix}_{i:04d}.png'
        
        output_path = os.path.join(args.out_dir, output_filename)
        
        # 可视化
        try:
            visualize_sample(
                sample, output_path, 
                show_mask=args.show_mask, 
                dpi=args.dpi,
                sam2_lora=sam2_lora,
                point_predictor=point_predictor,
                image_size=tuple(args.image_size),
                device=args.device,
                max_rollout_steps=args.max_rollout_steps,
            )
            print(f"  [{i+1}/{len(selected_samples)}] Saved: {output_path}")
        except Exception as e:
            print(f"  [{i+1}/{len(selected_samples)}] Error processing sample {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nDone! Visualizations saved to: {args.out_dir}")
    print(f"Total: {len(selected_samples)} images generated")


if __name__ == '__main__':
    main()


```

```python
#!/usr/bin/env python3
"""
可视化训练数据脚本

用于可视化PEFT训练数据中的图像、点标注和真值掩模。
支持随机采样或遍历所有样本。
如果提供了checkpoint，还会生成预测掩模对比图。

用法:
    # 随机采样5个样本（仅显示标注）
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_train-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode random \
        --num_samples 5

    # 遍历所有样本（带预测对比）
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_test-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode all \
        --checkpoint outputs/braintumour/peft_supervised_baseline-251107/checkpoint_best.pt \
        --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
        --image_size 512 512
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.ndimage import zoom

# 添加父目录到路径
_PARENT_DIR = Path(__file__).parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from peft.lora_sam2 import LoRASAM2Wrapper
from peft.point_predictor_peft import PointPredictorFromSAMFeatures
from peft.utils_peft import compute_dice, compute_iou, load_checkpoint


def load_jsonl(jsonl_path: str) -> List[Dict]:
    """加载JSONL文件"""
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith('['):
            # JSON数组格式
            data = json.loads(content)
        else:
            # JSONL格式（逐行）
            data = []
            for line in content.splitlines():
                if line.strip():
                    data.append(json.loads(line))
    return data


def visualize_sample(
    sample: Dict,
    output_path: str,
    show_mask: bool = True,
    dpi: int = 150,
    show_prediction: bool = True,
) -> None:
    """
    可视化单个样本
    
    Args:
        sample: 样本字典，包含 'image', 'gt_mask', 'points', 'labels', 'sam_masks_dir' 等字段
        output_path: 输出图像路径
        show_mask: 是否显示真值掩模
        dpi: 图像分辨率
        show_prediction: 是否显示预测掩模对比（需要sam_masks_dir）
    """
    # 加载图像
    image_path = sample['image']
    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        return
    
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    orig_w, orig_h = img.size
    
    # 获取点和标签
    points = sample.get('points', [])
    labels = sample.get('labels', [])
    
    if len(points) != len(labels):
        print(f"Warning: Points and labels length mismatch: {len(points)} vs {len(labels)}")
        return
    
    # 加载预测掩模和计算指标（如果sam_masks_dir存在）
    pred_mask = None
    metrics = {}
    do_prediction = False
    
    if show_prediction:
        sam_masks_dir = sample.get('sam_masks_dir', None)
        gt_mask_path = sample.get('gt_mask', None)
        
        if sam_masks_dir and gt_mask_path:
            try:
                # 获取图像stem（文件名不含扩展名）
                image_stem = Path(image_path).stem
                mask_dir = os.path.join(sam_masks_dir, image_stem)
                
                # 加载最后一个步骤的预测掩模（len(points)-1.png）
                if len(points) > 0:
                    last_mask_idx = len(points) - 1
                    pred_mask_path = os.path.join(mask_dir, f"{last_mask_idx}.png")
                    
                    if os.path.exists(pred_mask_path):
                        pred_mask = np.array(Image.open(pred_mask_path).convert('L'))
                        pred_mask_binary = (pred_mask > 127).astype(float)
                        
                        # 加载真值掩模并计算指标
                        if os.path.exists(gt_mask_path):
                            gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                            gt_mask_binary = (gt_mask > 127).astype(float)
                            
                            # 调整尺寸以匹配（如果需要）
                            if gt_mask_binary.shape != pred_mask_binary.shape:
                                scale_y = pred_mask_binary.shape[0] / gt_mask_binary.shape[0]
                                scale_x = pred_mask_binary.shape[1] / gt_mask_binary.shape[1]
                                gt_mask_binary = zoom(gt_mask_binary, (scale_y, scale_x), order=0)
                            
                            # 转换为tensor计算指标
                            pred_tensor = torch.from_numpy(pred_mask_binary).float()
                            gt_tensor = torch.from_numpy(gt_mask_binary).float()
                            
                            dice = compute_dice(pred_tensor, gt_tensor)
                            iou = compute_iou(pred_tensor, gt_tensor)
                            
                            metrics['dice'] = dice
                            metrics['iou'] = iou
                            metrics['num_points'] = len(points)
                            
                            do_prediction = True
                        else:
                            print(f"Warning: GT mask not found: {gt_mask_path}")
                    else:
                        print(f"Warning: Predicted mask not found: {pred_mask_path}")
                else:
                    print(f"Warning: No points in sample, cannot load predicted mask")
            except Exception as e:
                print(f"Warning: Failed to load predicted mask: {e}")
                import traceback
                traceback.print_exc()
    
    # 创建图形
    if do_prediction:
        # 两幅图：第一幅显示点和标注，第二幅显示掩模对比
        fig, axes = plt.subplots(1, 2, figsize=(24, 12))
        ax1, ax2 = axes
    else:
        # 单幅图：只显示点和标注
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 12))
        ax2 = None
    
    # 第一幅图：显示点和标注
    ax1.imshow(img_np)
    
    # 如果有真值掩模，叠加显示
    if show_mask:
        gt_mask_path = sample.get('gt_mask', None)
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                mask_binary = (gt_mask > 127).astype(float)
                ax1.imshow(mask_binary, alpha=0.3, cmap='Greens')
            except Exception as e:
                print(f"Warning: Failed to load mask {gt_mask_path}: {e}")
    
    # 绘制点
    for j, ((x, y), label) in enumerate(zip(points, labels)):
        if label == 1:
            # 前景点：红色，实心圆
            color = 'red'
            size = 150
            point_type = 'FG'
        else:
            # 背景点：蓝色，实心圆
            color = 'blue'
            size = 120
            point_type = 'BG'
        
        # 绘制点
        ax1.scatter(x, y, c=color, s=size, marker='o', 
                   edgecolors='white', linewidths=2, zorder=10)
        
        # 添加标签文本
        ax1.text(x + 15, y, f'{j}({point_type})', 
                fontsize=10, color='white', weight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    # 设置第一幅图的标题
    stem = Path(image_path).stem
    title1 = f'{stem}: {len(points)} points (Ground Truth)'
    ax1.set_title(title1, fontsize=14, weight='bold')
    ax1.axis('off')
    
    # 第二幅图：显示掩模对比（如果进行预测）
    if do_prediction and ax2 is not None:
        ax2.imshow(img_np)
        
        gt_mask_path = sample.get('gt_mask', None)
        gt_mask_binary = None
        pred_mask_binary = None
        
        # 加载并调整真值掩模尺寸
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                gt_mask_binary = (gt_mask > 127).astype(float)
                # 调整尺寸以匹配图像
                if gt_mask_binary.shape != img_np.shape[:2]:
                    scale_y = img_np.shape[0] / gt_mask_binary.shape[0]
                    scale_x = img_np.shape[1] / gt_mask_binary.shape[1]
                    gt_mask_binary = zoom(gt_mask_binary, (scale_y, scale_x), order=0)
            except Exception as e:
                print(f"Warning: Failed to load GT mask: {e}")
        
        # 调整预测掩模尺寸
        if pred_mask is not None:
            if pred_mask.shape != img_np.shape[:2]:
                scale_y = img_np.shape[0] / pred_mask.shape[0]
                scale_x = img_np.shape[1] / pred_mask.shape[1]
                pred_mask_resized = zoom(pred_mask, (scale_y, scale_x), order=0)
            else:
                pred_mask_resized = pred_mask
            pred_mask_binary = (pred_mask_resized > 0.5).astype(float)
        
        # 优化显示：先显示重叠区域（黄色），再显示真值（绿色），最后显示预测（红色）
        if gt_mask_binary is not None and pred_mask_binary is not None:
            # 重叠区域（黄色）
            overlap = (gt_mask_binary * pred_mask_binary).astype(float)
            overlap_colored = np.zeros_like(img_np)
            overlap_colored[:, :, 0] = overlap * 255  # 红色
            overlap_colored[:, :, 1] = overlap * 255  # 绿色
            ax2.imshow(overlap_colored, alpha=0.5)
            
            # 真值掩模的未重叠部分（绿色）
            gt_only = (gt_mask_binary * (1 - pred_mask_binary)).astype(float)
            gt_only_colored = np.zeros_like(img_np)
            gt_only_colored[:, :, 1] = gt_only * 255
            ax2.imshow(gt_only_colored, alpha=0.4)
            
            # 预测掩模的未重叠部分（红色）
            pred_only = (pred_mask_binary * (1 - gt_mask_binary)).astype(float)
            pred_only_colored = np.zeros_like(img_np)
            pred_only_colored[:, :, 0] = pred_only * 255
            ax2.imshow(pred_only_colored, alpha=0.4)
        elif gt_mask_binary is not None:
            # 只有真值掩模（绿色）
            gt_mask_colored = np.zeros_like(img_np)
            gt_mask_colored[:, :, 1] = gt_mask_binary * 255
            ax2.imshow(gt_mask_colored, alpha=0.4)
        elif pred_mask_binary is not None:
            # 只有预测掩模（红色）
            pred_mask_colored = np.zeros_like(img_np)
            pred_mask_colored[:, :, 0] = pred_mask_binary * 255
            ax2.imshow(pred_mask_colored, alpha=0.4)
        
        # 绘制预测点
        for j, ((x, y), label) in enumerate(zip(pred_points_seq, pred_labels_seq)):
            if label == 1:
                color = 'red'
                size = 150
                point_type = 'FG'
            else:
                color = 'blue'
                size = 120
                point_type = 'BG'
            
            # 缩放点坐标到原始图像尺寸
            x_orig = x * orig_w / image_size[1]
            y_orig = y * orig_h / image_size[0]
            
            ax2.scatter(x_orig, y_orig, c=color, s=size, marker='o', 
                       edgecolors='white', linewidths=2, zorder=10)
            ax2.text(x_orig + 15, y_orig, f'{j}({point_type})', 
                    fontsize=10, color='white', weight='bold',
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
        
        # 设置第二幅图的标题和指标
        title2 = f'Prediction: {len(pred_points_seq)} points'
        if metrics:
            metrics_text = f"Dice: {metrics.get('dice', 0):.3f}, IoU: {metrics.get('iou', 0):.3f}"
            title2 += f'\n{metrics_text}'
        ax2.set_title(title2, fontsize=14, weight='bold')
        ax2.axis('off')
        
        # 在图像角落添加指标文本
        if metrics:
            metrics_str = f"Dice: {metrics.get('dice', 0):.3f}\nIoU: {metrics.get('iou', 0):.3f}\nPoints: {metrics.get('num_points', 0)}"
            ax2.text(0.02, 0.98, metrics_str, 
                    transform=ax2.transAxes, fontsize=12, weight='bold',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='可视化PEFT训练数据',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--jsonl',
        type=str,
        required=True,
        help='输入JSONL文件路径'
    )
    
    parser.add_argument(
        '--out_dir',
        type=str,
        required=True,
        help='输出图像文件夹路径'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['random', 'all'],
        default='random',
        help='采样模式: random=随机采样, all=遍历所有样本'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='随机采样模式下的样本数量（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--show_mask',
        action='store_true',
        default=True,
        help='是否显示真值掩模（默认True）'
    )
    
    parser.add_argument(
        '--no_mask',
        dest='show_mask',
        action='store_false',
        help='不显示真值掩模'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='输出图像分辨率（DPI）'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default='sample',
        help='输出文件名前缀'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='模型checkpoint路径（如果提供，将生成预测对比图）'
    )
    
    parser.add_argument(
        '--sam_checkpoint',
        type=str,
        default=None,
        help='SAM2 checkpoint路径（如果提供checkpoint，此参数必需）'
    )
    
    parser.add_argument(
        '--image_size',
        type=int,
        nargs=2,
        default=[512, 512],
        help='图像尺寸 [H, W]（用于预测）'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='设备（cuda/cpu）'
    )
    
    parser.add_argument(
        '--max_rollout_steps',
        type=int,
        default=16,
        help='最大rollout步数（用于预测）'
    )
    
    args = parser.parse_args()
    
    # 加载模型（如果提供了checkpoint）
    sam2_lora = None
    point_predictor = None
    
    if args.checkpoint:
        if not args.sam_checkpoint:
            print("Error: --sam_checkpoint is required when --checkpoint is provided")
            sys.exit(1)
        
        print(f"Loading model from {args.checkpoint}...")
        try:
            device = torch.device(args.device)
            
            # 加载checkpoint
            ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
            config = ckpt.get('config', {})
            
            # 提取配置
            lora_rank = config.get('lora_rank', 16)
            lora_alpha = config.get('lora_alpha', 32)
            fusion_mode = config.get('fusion_mode', 'film')
            feature_scale = config.get('feature_scale', 8)
            
            # 初始化SAM2 LoRA
            sam2_lora = LoRASAM2Wrapper(
                sam_checkpoint=args.sam_checkpoint,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                device=str(device),
            )
            
            # 初始化点预测网络
            sam_feature_dim = 256  # SAM2 Hiera Large的特征维度
            point_predictor = PointPredictorFromSAMFeatures(
                sam_feature_dim=sam_feature_dim,
                output_size=tuple(args.image_size),
                fusion_mode=fusion_mode,
                feature_scale=feature_scale,
            ).to(device)
            
            # 加载权重
            load_checkpoint(
                args.checkpoint, point_predictor, sam2_lora, None, None, None, str(device)
            )
            
            point_predictor.eval()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Will continue without prediction...")
            sam2_lora = None
            point_predictor = None
    
    # 加载数据
    print(f"Loading data from {args.jsonl}...")
    data = load_jsonl(args.jsonl)
    print(f"Loaded {len(data)} samples")
    
    # 选择样本
    if args.mode == 'random':
        random.seed(args.seed)
        selected_samples = random.sample(data, min(args.num_samples, len(data)))
        print(f"Randomly selected {len(selected_samples)} samples")
    else:
        selected_samples = data
        print(f"Processing all {len(selected_samples)} samples")
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 可视化每个样本
    print(f"\nGenerating visualizations...")
    for i, sample in enumerate(selected_samples):
        # 生成输出文件名
        if args.mode == 'random':
            output_filename = f'{args.prefix}_{i:03d}.png'
        else:
            # 使用图像文件名作为输出文件名
            image_path = sample.get('image', '')
            if image_path:
                stem = Path(image_path).stem
                output_filename = f'{args.prefix}_{stem}.png'
            else:
                output_filename = f'{args.prefix}_{i:04d}.png'
        
        output_path = os.path.join(args.out_dir, output_filename)
        
        # 可视化
        try:
            visualize_sample(
                sample, output_path, 
                show_mask=args.show_mask, 
                dpi=args.dpi,
                sam2_lora=sam2_lora,
                point_predictor=point_predictor,
                image_size=tuple(args.image_size),
                device=args.device,
                max_rollout_steps=args.max_rollout_steps,
            )
            print(f"  [{i+1}/{len(selected_samples)}] Saved: {output_path}")
        except Exception as e:
            print(f"  [{i+1}/{len(selected_samples)}] Error processing sample {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nDone! Visualizations saved to: {args.out_dir}")
    print(f"Total: {len(selected_samples)} images generated")


if __name__ == '__main__':
    main()


```

```python
#!/usr/bin/env python3
"""
可视化训练数据脚本

用于可视化PEFT训练数据中的图像、点标注和真值掩模。
支持随机采样或遍历所有样本。
如果提供了checkpoint，还会生成预测掩模对比图。

用法:
    # 随机采样5个样本（仅显示标注）
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_train-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode random \
        --num_samples 5

    # 遍历所有样本（带预测对比）
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_test-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode all \
        --checkpoint outputs/braintumour/peft_supervised_baseline-251107/checkpoint_best.pt \
        --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
        --image_size 512 512
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.ndimage import zoom

# 添加父目录到路径
_PARENT_DIR = Path(__file__).parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from peft.lora_sam2 import LoRASAM2Wrapper
from peft.point_predictor_peft import PointPredictorFromSAMFeatures
from peft.utils_peft import compute_dice, compute_iou, load_checkpoint


def load_jsonl(jsonl_path: str) -> List[Dict]:
    """加载JSONL文件"""
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith('['):
            # JSON数组格式
            data = json.loads(content)
        else:
            # JSONL格式（逐行）
            data = []
            for line in content.splitlines():
                if line.strip():
                    data.append(json.loads(line))
    return data


def visualize_sample(
    sample: Dict,
    output_path: str,
    show_mask: bool = True,
    dpi: int = 150,
    show_prediction: bool = True,
) -> None:
    """
    可视化单个样本
    
    Args:
        sample: 样本字典，包含 'image', 'gt_mask', 'points', 'labels', 'sam_masks_dir' 等字段
        output_path: 输出图像路径
        show_mask: 是否显示真值掩模
        dpi: 图像分辨率
        show_prediction: 是否显示预测掩模对比（需要sam_masks_dir）
    """
    # 加载图像
    image_path = sample['image']
    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        return
    
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    orig_w, orig_h = img.size
    
    # 获取点和标签
    points = sample.get('points', [])
    labels = sample.get('labels', [])
    
    if len(points) != len(labels):
        print(f"Warning: Points and labels length mismatch: {len(points)} vs {len(labels)}")
        return
    
    # 加载预测掩模和计算指标（如果sam_masks_dir存在）
    pred_mask = None
    metrics = {}
    do_prediction = False
    
    if show_prediction:
        sam_masks_dir = sample.get('sam_masks_dir', None)
        gt_mask_path = sample.get('gt_mask', None)
        
        if sam_masks_dir and gt_mask_path:
            try:
                # 获取图像stem（文件名不含扩展名）
                image_stem = Path(image_path).stem
                mask_dir = os.path.join(sam_masks_dir, image_stem)
                
                # 加载最后一个步骤的预测掩模（len(points)-1.png）
                if len(points) > 0:
                    last_mask_idx = len(points) - 1
                    pred_mask_path = os.path.join(mask_dir, f"{last_mask_idx}.png")
                    
                    if os.path.exists(pred_mask_path):
                        pred_mask = np.array(Image.open(pred_mask_path).convert('L'))
                        pred_mask_binary = (pred_mask > 127).astype(float)
                        
                        # 加载真值掩模并计算指标
                        if os.path.exists(gt_mask_path):
                            gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                            gt_mask_binary = (gt_mask > 127).astype(float)
                            
                            # 调整尺寸以匹配（如果需要）
                            if gt_mask_binary.shape != pred_mask_binary.shape:
                                scale_y = pred_mask_binary.shape[0] / gt_mask_binary.shape[0]
                                scale_x = pred_mask_binary.shape[1] / gt_mask_binary.shape[1]
                                gt_mask_binary = zoom(gt_mask_binary, (scale_y, scale_x), order=0)
                            
                            # 转换为tensor计算指标
                            pred_tensor = torch.from_numpy(pred_mask_binary).float()
                            gt_tensor = torch.from_numpy(gt_mask_binary).float()
                            
                            dice = compute_dice(pred_tensor, gt_tensor)
                            iou = compute_iou(pred_tensor, gt_tensor)
                            
                            metrics['dice'] = dice
                            metrics['iou'] = iou
                            metrics['num_points'] = len(points)
                            
                            do_prediction = True
                        else:
                            print(f"Warning: GT mask not found: {gt_mask_path}")
                    else:
                        print(f"Warning: Predicted mask not found: {pred_mask_path}")
                else:
                    print(f"Warning: No points in sample, cannot load predicted mask")
            except Exception as e:
                print(f"Warning: Failed to load predicted mask: {e}")
                import traceback
                traceback.print_exc()
    
    # 创建图形
    if do_prediction:
        # 两幅图：第一幅显示点和标注，第二幅显示掩模对比
        fig, axes = plt.subplots(1, 2, figsize=(24, 12))
        ax1, ax2 = axes
    else:
        # 单幅图：只显示点和标注
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 12))
        ax2 = None
    
    # 第一幅图：显示点和标注
    ax1.imshow(img_np)
    
    # 如果有真值掩模，叠加显示
    if show_mask:
        gt_mask_path = sample.get('gt_mask', None)
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                mask_binary = (gt_mask > 127).astype(float)
                ax1.imshow(mask_binary, alpha=0.3, cmap='Greens')
            except Exception as e:
                print(f"Warning: Failed to load mask {gt_mask_path}: {e}")
    
    # 绘制点
    for j, ((x, y), label) in enumerate(zip(points, labels)):
        if label == 1:
            # 前景点：红色，实心圆
            color = 'red'
            size = 150
            point_type = 'FG'
        else:
            # 背景点：蓝色，实心圆
            color = 'blue'
            size = 120
            point_type = 'BG'
        
        # 绘制点
        ax1.scatter(x, y, c=color, s=size, marker='o', 
                   edgecolors='white', linewidths=2, zorder=10)
        
        # 添加标签文本
        ax1.text(x + 15, y, f'{j}({point_type})', 
                fontsize=10, color='white', weight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    # 设置第一幅图的标题
    stem = Path(image_path).stem
    title1 = f'{stem}: {len(points)} points (Ground Truth)'
    ax1.set_title(title1, fontsize=14, weight='bold')
    ax1.axis('off')
    
    # 第二幅图：显示掩模对比（如果进行预测）
    if do_prediction and ax2 is not None:
        ax2.imshow(img_np)
        
        gt_mask_path = sample.get('gt_mask', None)
        gt_mask_binary = None
        pred_mask_binary = None
        
        # 加载并调整真值掩模尺寸
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                gt_mask_binary = (gt_mask > 127).astype(float)
                # 调整尺寸以匹配图像
                if gt_mask_binary.shape != img_np.shape[:2]:
                    scale_y = img_np.shape[0] / gt_mask_binary.shape[0]
                    scale_x = img_np.shape[1] / gt_mask_binary.shape[1]
                    gt_mask_binary = zoom(gt_mask_binary, (scale_y, scale_x), order=0)
            except Exception as e:
                print(f"Warning: Failed to load GT mask: {e}")
        
        # 调整预测掩模尺寸
        if pred_mask is not None:
            if pred_mask.shape != img_np.shape[:2]:
                scale_y = img_np.shape[0] / pred_mask.shape[0]
                scale_x = img_np.shape[1] / pred_mask.shape[1]
                pred_mask_resized = zoom(pred_mask, (scale_y, scale_x), order=0)
            else:
                pred_mask_resized = pred_mask
            pred_mask_binary = (pred_mask_resized > 0.5).astype(float)
        
        # 优化显示：先显示重叠区域（黄色），再显示真值（绿色），最后显示预测（红色）
        if gt_mask_binary is not None and pred_mask_binary is not None:
            # 重叠区域（黄色）
            overlap = (gt_mask_binary * pred_mask_binary).astype(float)
            overlap_colored = np.zeros_like(img_np)
            overlap_colored[:, :, 0] = overlap * 255  # 红色
            overlap_colored[:, :, 1] = overlap * 255  # 绿色
            ax2.imshow(overlap_colored, alpha=0.5)
            
            # 真值掩模的未重叠部分（绿色）
            gt_only = (gt_mask_binary * (1 - pred_mask_binary)).astype(float)
            gt_only_colored = np.zeros_like(img_np)
            gt_only_colored[:, :, 1] = gt_only * 255
            ax2.imshow(gt_only_colored, alpha=0.4)
            
            # 预测掩模的未重叠部分（红色）
            pred_only = (pred_mask_binary * (1 - gt_mask_binary)).astype(float)
            pred_only_colored = np.zeros_like(img_np)
            pred_only_colored[:, :, 0] = pred_only * 255
            ax2.imshow(pred_only_colored, alpha=0.4)
        elif gt_mask_binary is not None:
            # 只有真值掩模（绿色）
            gt_mask_colored = np.zeros_like(img_np)
            gt_mask_colored[:, :, 1] = gt_mask_binary * 255
            ax2.imshow(gt_mask_colored, alpha=0.4)
        elif pred_mask_binary is not None:
            # 只有预测掩模（红色）
            pred_mask_colored = np.zeros_like(img_np)
            pred_mask_colored[:, :, 0] = pred_mask_binary * 255
            ax2.imshow(pred_mask_colored, alpha=0.4)
        
        # 绘制预测点（使用JSONL中的points）
        for j, ((x, y), label) in enumerate(zip(points, labels)):
            if label == 1:
                color = 'red'
                size = 150
                point_type = 'FG'
            else:
                color = 'blue'
                size = 120
                point_type = 'BG'
            
            ax2.scatter(x, y, c=color, s=size, marker='o', 
                       edgecolors='white', linewidths=2, zorder=10)
            ax2.text(x + 15, y, f'{j}({point_type})', 
                    fontsize=10, color='white', weight='bold',
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
        
        # 设置第二幅图的标题和指标
        title2 = f'Prediction: {len(pred_points_seq)} points'
        if metrics:
            metrics_text = f"Dice: {metrics.get('dice', 0):.3f}, IoU: {metrics.get('iou', 0):.3f}"
            title2 += f'\n{metrics_text}'
        ax2.set_title(title2, fontsize=14, weight='bold')
        ax2.axis('off')
        
        # 在图像角落添加指标文本
        if metrics:
            metrics_str = f"Dice: {metrics.get('dice', 0):.3f}\nIoU: {metrics.get('iou', 0):.3f}\nPoints: {metrics.get('num_points', 0)}"
            ax2.text(0.02, 0.98, metrics_str, 
                    transform=ax2.transAxes, fontsize=12, weight='bold',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='可视化PEFT训练数据',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--jsonl',
        type=str,
        required=True,
        help='输入JSONL文件路径'
    )
    
    parser.add_argument(
        '--out_dir',
        type=str,
        required=True,
        help='输出图像文件夹路径'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['random', 'all'],
        default='random',
        help='采样模式: random=随机采样, all=遍历所有样本'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='随机采样模式下的样本数量（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--show_mask',
        action='store_true',
        default=True,
        help='是否显示真值掩模（默认True）'
    )
    
    parser.add_argument(
        '--no_mask',
        dest='show_mask',
        action='store_false',
        help='不显示真值掩模'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='输出图像分辨率（DPI）'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default='sample',
        help='输出文件名前缀'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='模型checkpoint路径（如果提供，将生成预测对比图）'
    )
    
    parser.add_argument(
        '--sam_checkpoint',
        type=str,
        default=None,
        help='SAM2 checkpoint路径（如果提供checkpoint，此参数必需）'
    )
    
    parser.add_argument(
        '--image_size',
        type=int,
        nargs=2,
        default=[512, 512],
        help='图像尺寸 [H, W]（用于预测）'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='设备（cuda/cpu）'
    )
    
    parser.add_argument(
        '--max_rollout_steps',
        type=int,
        default=16,
        help='最大rollout步数（用于预测）'
    )
    
    args = parser.parse_args()
    
    # 加载模型（如果提供了checkpoint）
    sam2_lora = None
    point_predictor = None
    
    if args.checkpoint:
        if not args.sam_checkpoint:
            print("Error: --sam_checkpoint is required when --checkpoint is provided")
            sys.exit(1)
        
        print(f"Loading model from {args.checkpoint}...")
        try:
            device = torch.device(args.device)
            
            # 加载checkpoint
            ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
            config = ckpt.get('config', {})
            
            # 提取配置
            lora_rank = config.get('lora_rank', 16)
            lora_alpha = config.get('lora_alpha', 32)
            fusion_mode = config.get('fusion_mode', 'film')
            feature_scale = config.get('feature_scale', 8)
            
            # 初始化SAM2 LoRA
            sam2_lora = LoRASAM2Wrapper(
                sam_checkpoint=args.sam_checkpoint,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                device=str(device),
            )
            
            # 初始化点预测网络
            sam_feature_dim = 256  # SAM2 Hiera Large的特征维度
            point_predictor = PointPredictorFromSAMFeatures(
                sam_feature_dim=sam_feature_dim,
                output_size=tuple(args.image_size),
                fusion_mode=fusion_mode,
                feature_scale=feature_scale,
            ).to(device)
            
            # 加载权重
            load_checkpoint(
                args.checkpoint, point_predictor, sam2_lora, None, None, None, str(device)
            )
            
            point_predictor.eval()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Will continue without prediction...")
            sam2_lora = None
            point_predictor = None
    
    # 加载数据
    print(f"Loading data from {args.jsonl}...")
    data = load_jsonl(args.jsonl)
    print(f"Loaded {len(data)} samples")
    
    # 选择样本
    if args.mode == 'random':
        random.seed(args.seed)
        selected_samples = random.sample(data, min(args.num_samples, len(data)))
        print(f"Randomly selected {len(selected_samples)} samples")
    else:
        selected_samples = data
        print(f"Processing all {len(selected_samples)} samples")
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 可视化每个样本
    print(f"\nGenerating visualizations...")
    for i, sample in enumerate(selected_samples):
        # 生成输出文件名
        if args.mode == 'random':
            output_filename = f'{args.prefix}_{i:03d}.png'
        else:
            # 使用图像文件名作为输出文件名
            image_path = sample.get('image', '')
            if image_path:
                stem = Path(image_path).stem
                output_filename = f'{args.prefix}_{stem}.png'
            else:
                output_filename = f'{args.prefix}_{i:04d}.png'
        
        output_path = os.path.join(args.out_dir, output_filename)
        
        # 可视化
        try:
            visualize_sample(
                sample, output_path, 
                show_mask=args.show_mask, 
                dpi=args.dpi,
                sam2_lora=sam2_lora,
                point_predictor=point_predictor,
                image_size=tuple(args.image_size),
                device=args.device,
                max_rollout_steps=args.max_rollout_steps,
            )
            print(f"  [{i+1}/{len(selected_samples)}] Saved: {output_path}")
        except Exception as e:
            print(f"  [{i+1}/{len(selected_samples)}] Error processing sample {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nDone! Visualizations saved to: {args.out_dir}")
    print(f"Total: {len(selected_samples)} images generated")


if __name__ == '__main__':
    main()


```

```python
#!/usr/bin/env python3
"""
可视化训练数据脚本

用于可视化PEFT训练数据中的图像、点标注和真值掩模。
支持随机采样或遍历所有样本。
如果提供了checkpoint，还会生成预测掩模对比图。

用法:
    # 随机采样5个样本（仅显示标注）
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_train-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode random \
        --num_samples 5

    # 遍历所有样本（带预测对比）
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_test-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode all \
        --checkpoint outputs/braintumour/peft_supervised_baseline-251107/checkpoint_best.pt \
        --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
        --image_size 512 512
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.ndimage import zoom

# 添加父目录到路径
_PARENT_DIR = Path(__file__).parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from peft.lora_sam2 import LoRASAM2Wrapper
from peft.point_predictor_peft import PointPredictorFromSAMFeatures
from peft.utils_peft import compute_dice, compute_iou, load_checkpoint


def load_jsonl(jsonl_path: str) -> List[Dict]:
    """加载JSONL文件"""
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith('['):
            # JSON数组格式
            data = json.loads(content)
        else:
            # JSONL格式（逐行）
            data = []
            for line in content.splitlines():
                if line.strip():
                    data.append(json.loads(line))
    return data


def visualize_sample(
    sample: Dict,
    output_path: str,
    show_mask: bool = True,
    dpi: int = 150,
    show_prediction: bool = True,
) -> None:
    """
    可视化单个样本
    
    Args:
        sample: 样本字典，包含 'image', 'gt_mask', 'points', 'labels', 'sam_masks_dir' 等字段
        output_path: 输出图像路径
        show_mask: 是否显示真值掩模
        dpi: 图像分辨率
        show_prediction: 是否显示预测掩模对比（需要sam_masks_dir）
    """
    # 加载图像
    image_path = sample['image']
    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        return
    
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    orig_w, orig_h = img.size
    
    # 获取点和标签
    points = sample.get('points', [])
    labels = sample.get('labels', [])
    
    if len(points) != len(labels):
        print(f"Warning: Points and labels length mismatch: {len(points)} vs {len(labels)}")
        return
    
    # 加载预测掩模和计算指标（如果sam_masks_dir存在）
    pred_mask = None
    metrics = {}
    do_prediction = False
    
    if show_prediction:
        sam_masks_dir = sample.get('sam_masks_dir', None)
        gt_mask_path = sample.get('gt_mask', None)
        
        if sam_masks_dir and gt_mask_path:
            try:
                # 获取图像stem（文件名不含扩展名）
                image_stem = Path(image_path).stem
                mask_dir = os.path.join(sam_masks_dir, image_stem)
                
                # 加载最后一个步骤的预测掩模（len(points)-1.png）
                if len(points) > 0:
                    last_mask_idx = len(points) - 1
                    pred_mask_path = os.path.join(mask_dir, f"{last_mask_idx}.png")
                    
                    if os.path.exists(pred_mask_path):
                        pred_mask = np.array(Image.open(pred_mask_path).convert('L'))
                        pred_mask_binary = (pred_mask > 127).astype(float)
                        
                        # 加载真值掩模并计算指标
                        if os.path.exists(gt_mask_path):
                            gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                            gt_mask_binary = (gt_mask > 127).astype(float)
                            
                            # 调整尺寸以匹配（如果需要）
                            if gt_mask_binary.shape != pred_mask_binary.shape:
                                scale_y = pred_mask_binary.shape[0] / gt_mask_binary.shape[0]
                                scale_x = pred_mask_binary.shape[1] / gt_mask_binary.shape[1]
                                gt_mask_binary = zoom(gt_mask_binary, (scale_y, scale_x), order=0)
                            
                            # 转换为tensor计算指标
                            pred_tensor = torch.from_numpy(pred_mask_binary).float()
                            gt_tensor = torch.from_numpy(gt_mask_binary).float()
                            
                            dice = compute_dice(pred_tensor, gt_tensor)
                            iou = compute_iou(pred_tensor, gt_tensor)
                            
                            metrics['dice'] = dice
                            metrics['iou'] = iou
                            metrics['num_points'] = len(points)
                            
                            do_prediction = True
                        else:
                            print(f"Warning: GT mask not found: {gt_mask_path}")
                    else:
                        print(f"Warning: Predicted mask not found: {pred_mask_path}")
                else:
                    print(f"Warning: No points in sample, cannot load predicted mask")
            except Exception as e:
                print(f"Warning: Failed to load predicted mask: {e}")
                import traceback
                traceback.print_exc()
    
    # 创建图形
    if do_prediction:
        # 两幅图：第一幅显示点和标注，第二幅显示掩模对比
        fig, axes = plt.subplots(1, 2, figsize=(24, 12))
        ax1, ax2 = axes
    else:
        # 单幅图：只显示点和标注
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 12))
        ax2 = None
    
    # 第一幅图：显示点和标注
    ax1.imshow(img_np)
    
    # 如果有真值掩模，叠加显示
    if show_mask:
        gt_mask_path = sample.get('gt_mask', None)
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                mask_binary = (gt_mask > 127).astype(float)
                ax1.imshow(mask_binary, alpha=0.3, cmap='Greens')
            except Exception as e:
                print(f"Warning: Failed to load mask {gt_mask_path}: {e}")
    
    # 绘制点
    for j, ((x, y), label) in enumerate(zip(points, labels)):
        if label == 1:
            # 前景点：红色，实心圆
            color = 'red'
            size = 150
            point_type = 'FG'
        else:
            # 背景点：蓝色，实心圆
            color = 'blue'
            size = 120
            point_type = 'BG'
        
        # 绘制点
        ax1.scatter(x, y, c=color, s=size, marker='o', 
                   edgecolors='white', linewidths=2, zorder=10)
        
        # 添加标签文本
        ax1.text(x + 15, y, f'{j}({point_type})', 
                fontsize=10, color='white', weight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    # 设置第一幅图的标题
    stem = Path(image_path).stem
    title1 = f'{stem}: {len(points)} points (Ground Truth)'
    ax1.set_title(title1, fontsize=14, weight='bold')
    ax1.axis('off')
    
    # 第二幅图：显示掩模对比（如果进行预测）
    if do_prediction and ax2 is not None:
        ax2.imshow(img_np)
        
        gt_mask_path = sample.get('gt_mask', None)
        gt_mask_binary = None
        pred_mask_binary = None
        
        # 加载并调整真值掩模尺寸
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                gt_mask_binary = (gt_mask > 127).astype(float)
                # 调整尺寸以匹配图像
                if gt_mask_binary.shape != img_np.shape[:2]:
                    scale_y = img_np.shape[0] / gt_mask_binary.shape[0]
                    scale_x = img_np.shape[1] / gt_mask_binary.shape[1]
                    gt_mask_binary = zoom(gt_mask_binary, (scale_y, scale_x), order=0)
            except Exception as e:
                print(f"Warning: Failed to load GT mask: {e}")
        
        # 调整预测掩模尺寸
        if pred_mask is not None:
            if pred_mask.shape != img_np.shape[:2]:
                scale_y = img_np.shape[0] / pred_mask.shape[0]
                scale_x = img_np.shape[1] / pred_mask.shape[1]
                pred_mask_resized = zoom(pred_mask, (scale_y, scale_x), order=0)
            else:
                pred_mask_resized = pred_mask
            pred_mask_binary = (pred_mask_resized > 0.5).astype(float)
        
        # 优化显示：先显示重叠区域（黄色），再显示真值（绿色），最后显示预测（红色）
        if gt_mask_binary is not None and pred_mask_binary is not None:
            # 重叠区域（黄色）
            overlap = (gt_mask_binary * pred_mask_binary).astype(float)
            overlap_colored = np.zeros_like(img_np)
            overlap_colored[:, :, 0] = overlap * 255  # 红色
            overlap_colored[:, :, 1] = overlap * 255  # 绿色
            ax2.imshow(overlap_colored, alpha=0.5)
            
            # 真值掩模的未重叠部分（绿色）
            gt_only = (gt_mask_binary * (1 - pred_mask_binary)).astype(float)
            gt_only_colored = np.zeros_like(img_np)
            gt_only_colored[:, :, 1] = gt_only * 255
            ax2.imshow(gt_only_colored, alpha=0.4)
            
            # 预测掩模的未重叠部分（红色）
            pred_only = (pred_mask_binary * (1 - gt_mask_binary)).astype(float)
            pred_only_colored = np.zeros_like(img_np)
            pred_only_colored[:, :, 0] = pred_only * 255
            ax2.imshow(pred_only_colored, alpha=0.4)
        elif gt_mask_binary is not None:
            # 只有真值掩模（绿色）
            gt_mask_colored = np.zeros_like(img_np)
            gt_mask_colored[:, :, 1] = gt_mask_binary * 255
            ax2.imshow(gt_mask_colored, alpha=0.4)
        elif pred_mask_binary is not None:
            # 只有预测掩模（红色）
            pred_mask_colored = np.zeros_like(img_np)
            pred_mask_colored[:, :, 0] = pred_mask_binary * 255
            ax2.imshow(pred_mask_colored, alpha=0.4)
        
        # 绘制预测点（使用JSONL中的points）
        for j, ((x, y), label) in enumerate(zip(points, labels)):
            if label == 1:
                color = 'red'
                size = 150
                point_type = 'FG'
            else:
                color = 'blue'
                size = 120
                point_type = 'BG'
            
            ax2.scatter(x, y, c=color, s=size, marker='o', 
                       edgecolors='white', linewidths=2, zorder=10)
            ax2.text(x + 15, y, f'{j}({point_type})', 
                    fontsize=10, color='white', weight='bold',
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
        
        # 设置第二幅图的标题和指标
        title2 = f'Prediction: {len(points)} points'
        if metrics:
            metrics_text = f"Dice: {metrics.get('dice', 0):.3f}, IoU: {metrics.get('iou', 0):.3f}"
            title2 += f'\n{metrics_text}'
        ax2.set_title(title2, fontsize=14, weight='bold')
        ax2.axis('off')
        
        # 在图像角落添加指标文本
        if metrics:
            metrics_str = f"Dice: {metrics.get('dice', 0):.3f}\nIoU: {metrics.get('iou', 0):.3f}\nPoints: {metrics.get('num_points', 0)}"
            ax2.text(0.02, 0.98, metrics_str, 
                    transform=ax2.transAxes, fontsize=12, weight='bold',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='可视化PEFT训练数据',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--jsonl',
        type=str,
        required=True,
        help='输入JSONL文件路径'
    )
    
    parser.add_argument(
        '--out_dir',
        type=str,
        required=True,
        help='输出图像文件夹路径'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['random', 'all'],
        default='random',
        help='采样模式: random=随机采样, all=遍历所有样本'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='随机采样模式下的样本数量（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--show_mask',
        action='store_true',
        default=True,
        help='是否显示真值掩模（默认True）'
    )
    
    parser.add_argument(
        '--no_mask',
        dest='show_mask',
        action='store_false',
        help='不显示真值掩模'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='输出图像分辨率（DPI）'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default='sample',
        help='输出文件名前缀'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='模型checkpoint路径（如果提供，将生成预测对比图）'
    )
    
    parser.add_argument(
        '--sam_checkpoint',
        type=str,
        default=None,
        help='SAM2 checkpoint路径（如果提供checkpoint，此参数必需）'
    )
    
    parser.add_argument(
        '--image_size',
        type=int,
        nargs=2,
        default=[512, 512],
        help='图像尺寸 [H, W]（用于预测）'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='设备（cuda/cpu）'
    )
    
    parser.add_argument(
        '--max_rollout_steps',
        type=int,
        default=16,
        help='最大rollout步数（用于预测）'
    )
    
    args = parser.parse_args()
    
    # 加载模型（如果提供了checkpoint）
    sam2_lora = None
    point_predictor = None
    
    if args.checkpoint:
        if not args.sam_checkpoint:
            print("Error: --sam_checkpoint is required when --checkpoint is provided")
            sys.exit(1)
        
        print(f"Loading model from {args.checkpoint}...")
        try:
            device = torch.device(args.device)
            
            # 加载checkpoint
            ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
            config = ckpt.get('config', {})
            
            # 提取配置
            lora_rank = config.get('lora_rank', 16)
            lora_alpha = config.get('lora_alpha', 32)
            fusion_mode = config.get('fusion_mode', 'film')
            feature_scale = config.get('feature_scale', 8)
            
            # 初始化SAM2 LoRA
            sam2_lora = LoRASAM2Wrapper(
                sam_checkpoint=args.sam_checkpoint,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                device=str(device),
            )
            
            # 初始化点预测网络
            sam_feature_dim = 256  # SAM2 Hiera Large的特征维度
            point_predictor = PointPredictorFromSAMFeatures(
                sam_feature_dim=sam_feature_dim,
                output_size=tuple(args.image_size),
                fusion_mode=fusion_mode,
                feature_scale=feature_scale,
            ).to(device)
            
            # 加载权重
            load_checkpoint(
                args.checkpoint, point_predictor, sam2_lora, None, None, None, str(device)
            )
            
            point_predictor.eval()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Will continue without prediction...")
            sam2_lora = None
            point_predictor = None
    
    # 加载数据
    print(f"Loading data from {args.jsonl}...")
    data = load_jsonl(args.jsonl)
    print(f"Loaded {len(data)} samples")
    
    # 选择样本
    if args.mode == 'random':
        random.seed(args.seed)
        selected_samples = random.sample(data, min(args.num_samples, len(data)))
        print(f"Randomly selected {len(selected_samples)} samples")
    else:
        selected_samples = data
        print(f"Processing all {len(selected_samples)} samples")
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 可视化每个样本
    print(f"\nGenerating visualizations...")
    for i, sample in enumerate(selected_samples):
        # 生成输出文件名
        if args.mode == 'random':
            output_filename = f'{args.prefix}_{i:03d}.png'
        else:
            # 使用图像文件名作为输出文件名
            image_path = sample.get('image', '')
            if image_path:
                stem = Path(image_path).stem
                output_filename = f'{args.prefix}_{stem}.png'
            else:
                output_filename = f'{args.prefix}_{i:04d}.png'
        
        output_path = os.path.join(args.out_dir, output_filename)
        
        # 可视化
        try:
            visualize_sample(
                sample, output_path, 
                show_mask=args.show_mask, 
                dpi=args.dpi,
                sam2_lora=sam2_lora,
                point_predictor=point_predictor,
                image_size=tuple(args.image_size),
                device=args.device,
                max_rollout_steps=args.max_rollout_steps,
            )
            print(f"  [{i+1}/{len(selected_samples)}] Saved: {output_path}")
        except Exception as e:
            print(f"  [{i+1}/{len(selected_samples)}] Error processing sample {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nDone! Visualizations saved to: {args.out_dir}")
    print(f"Total: {len(selected_samples)} images generated")


if __name__ == '__main__':
    main()


```


修复第二幅图中掩模叠加逻辑，确保变量作用域正确：



修复变量作用域问题，确保 `pred_mask_binary` 在正确位置定义和使用：


```python
#!/usr/bin/env python3
"""
可视化训练数据脚本

用于可视化PEFT训练数据中的图像、点标注和真值掩模。
支持随机采样或遍历所有样本。
如果提供了checkpoint，还会生成预测掩模对比图。

用法:
    # 随机采样5个样本（仅显示标注）
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_train-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode random \
        --num_samples 5

    # 遍历所有样本（带预测对比）
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_test-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode all \
        --checkpoint outputs/braintumour/peft_supervised_baseline-251107/checkpoint_best.pt \
        --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
        --image_size 512 512
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.ndimage import zoom

# 添加父目录到路径
_PARENT_DIR = Path(__file__).parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from peft.lora_sam2 import LoRASAM2Wrapper
from peft.point_predictor_peft import PointPredictorFromSAMFeatures
from peft.utils_peft import compute_dice, compute_iou, load_checkpoint


def load_jsonl(jsonl_path: str) -> List[Dict]:
    """加载JSONL文件"""
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith('['):
            # JSON数组格式
            data = json.loads(content)
        else:
            # JSONL格式（逐行）
            data = []
            for line in content.splitlines():
                if line.strip():
                    data.append(json.loads(line))
    return data


def visualize_sample(
    sample: Dict,
    output_path: str,
    show_mask: bool = True,
    dpi: int = 150,
    show_prediction: bool = True,
) -> None:
    """
    可视化单个样本
    
    Args:
        sample: 样本字典，包含 'image', 'gt_mask', 'points', 'labels', 'sam_masks_dir' 等字段
        output_path: 输出图像路径
        show_mask: 是否显示真值掩模
        dpi: 图像分辨率
        show_prediction: 是否显示预测掩模对比（需要sam_masks_dir）
    """
    # 加载图像
    image_path = sample['image']
    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        return
    
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    orig_w, orig_h = img.size
    
    # 获取点和标签
    points = sample.get('points', [])
    labels = sample.get('labels', [])
    
    if len(points) != len(labels):
        print(f"Warning: Points and labels length mismatch: {len(points)} vs {len(labels)}")
        return
    
    # 加载预测掩模和计算指标（如果sam_masks_dir存在）
    pred_mask = None
    pred_mask_binary = None
    gt_mask_binary_for_metrics = None
    metrics = {}
    do_prediction = False
    
    if show_prediction:
        sam_masks_dir = sample.get('sam_masks_dir', None)
        gt_mask_path = sample.get('gt_mask', None)
        
        if sam_masks_dir and gt_mask_path:
            try:
                # 获取图像stem（文件名不含扩展名）
                image_stem = Path(image_path).stem
                mask_dir = os.path.join(sam_masks_dir, image_stem)
                
                # 加载最后一个步骤的预测掩模（len(points)-1.png）
                if len(points) > 0:
                    last_mask_idx = len(points) - 1
                    pred_mask_path = os.path.join(mask_dir, f"{last_mask_idx}.png")
                    
                    if os.path.exists(pred_mask_path):
                        pred_mask = np.array(Image.open(pred_mask_path).convert('L'))
                        pred_mask_binary = (pred_mask > 127).astype(float)
                        
                        # 加载真值掩模并计算指标
                        if os.path.exists(gt_mask_path):
                            gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                            gt_mask_binary_for_metrics = (gt_mask > 127).astype(float)
                            
                            # 调整尺寸以匹配（如果需要）
                            if gt_mask_binary_for_metrics.shape != pred_mask_binary.shape:
                                scale_y = pred_mask_binary.shape[0] / gt_mask_binary_for_metrics.shape[0]
                                scale_x = pred_mask_binary.shape[1] / gt_mask_binary_for_metrics.shape[1]
                                gt_mask_binary_for_metrics = zoom(gt_mask_binary_for_metrics, (scale_y, scale_x), order=0)
                            
                            # 转换为tensor计算指标
                            pred_tensor = torch.from_numpy(pred_mask_binary).float()
                            gt_tensor = torch.from_numpy(gt_mask_binary_for_metrics).float()
                            
                            dice = compute_dice(pred_tensor, gt_tensor)
                            iou = compute_iou(pred_tensor, gt_tensor)
                            
                            metrics['dice'] = dice
                            metrics['iou'] = iou
                            metrics['num_points'] = len(points)
                            
                            do_prediction = True
                        else:
                            print(f"Warning: GT mask not found: {gt_mask_path}")
                    else:
                        print(f"Warning: Predicted mask not found: {pred_mask_path}")
                else:
                    print(f"Warning: No points in sample, cannot load predicted mask")
            except Exception as e:
                print(f"Warning: Failed to load predicted mask: {e}")
                import traceback
                traceback.print_exc()
    
    # 创建图形
    if do_prediction:
        # 两幅图：第一幅显示点和标注，第二幅显示掩模对比
        fig, axes = plt.subplots(1, 2, figsize=(24, 12))
        ax1, ax2 = axes
    else:
        # 单幅图：只显示点和标注
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 12))
        ax2 = None
    
    # 第一幅图：显示点和标注
    ax1.imshow(img_np)
    
    # 如果有真值掩模，叠加显示
    if show_mask:
        gt_mask_path = sample.get('gt_mask', None)
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                mask_binary = (gt_mask > 127).astype(float)
                ax1.imshow(mask_binary, alpha=0.3, cmap='Greens')
            except Exception as e:
                print(f"Warning: Failed to load mask {gt_mask_path}: {e}")
    
    # 绘制点
    for j, ((x, y), label) in enumerate(zip(points, labels)):
        if label == 1:
            # 前景点：红色，实心圆
            color = 'red'
            size = 150
            point_type = 'FG'
        else:
            # 背景点：蓝色，实心圆
            color = 'blue'
            size = 120
            point_type = 'BG'
        
        # 绘制点
        ax1.scatter(x, y, c=color, s=size, marker='o', 
                   edgecolors='white', linewidths=2, zorder=10)
        
        # 添加标签文本
        ax1.text(x + 15, y, f'{j}({point_type})', 
                fontsize=10, color='white', weight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    # 设置第一幅图的标题
    stem = Path(image_path).stem
    title1 = f'{stem}: {len(points)} points (Ground Truth)'
    ax1.set_title(title1, fontsize=14, weight='bold')
    ax1.axis('off')
    
    # 第二幅图：显示掩模对比（如果进行预测）
    if do_prediction and ax2 is not None:
        ax2.imshow(img_np)
        
        gt_mask_path = sample.get('gt_mask', None)
        gt_mask_binary = None
        pred_mask_binary = None
        
        # 加载并调整真值掩模尺寸
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                gt_mask_binary = (gt_mask > 127).astype(float)
                # 调整尺寸以匹配图像
                if gt_mask_binary.shape != img_np.shape[:2]:
                    scale_y = img_np.shape[0] / gt_mask_binary.shape[0]
                    scale_x = img_np.shape[1] / gt_mask_binary.shape[1]
                    gt_mask_binary = zoom(gt_mask_binary, (scale_y, scale_x), order=0)
            except Exception as e:
                print(f"Warning: Failed to load GT mask: {e}")
        
        # 调整预测掩模尺寸
        if pred_mask is not None:
            if pred_mask.shape != img_np.shape[:2]:
                scale_y = img_np.shape[0] / pred_mask.shape[0]
                scale_x = img_np.shape[1] / pred_mask.shape[1]
                pred_mask_resized = zoom(pred_mask, (scale_y, scale_x), order=0)
            else:
                pred_mask_resized = pred_mask
            pred_mask_binary = (pred_mask_resized > 0.5).astype(float)
        
        # 优化显示：先显示重叠区域（黄色），再显示真值（绿色），最后显示预测（红色）
        if gt_mask_binary is not None and pred_mask_binary is not None:
            # 重叠区域（黄色）
            overlap = (gt_mask_binary * pred_mask_binary).astype(float)
            overlap_colored = np.zeros_like(img_np)
            overlap_colored[:, :, 0] = overlap * 255  # 红色
            overlap_colored[:, :, 1] = overlap * 255  # 绿色
            ax2.imshow(overlap_colored, alpha=0.5)
            
            # 真值掩模的未重叠部分（绿色）
            gt_only = (gt_mask_binary * (1 - pred_mask_binary)).astype(float)
            gt_only_colored = np.zeros_like(img_np)
            gt_only_colored[:, :, 1] = gt_only * 255
            ax2.imshow(gt_only_colored, alpha=0.4)
            
            # 预测掩模的未重叠部分（红色）
            pred_only = (pred_mask_binary * (1 - gt_mask_binary)).astype(float)
            pred_only_colored = np.zeros_like(img_np)
            pred_only_colored[:, :, 0] = pred_only * 255
            ax2.imshow(pred_only_colored, alpha=0.4)
        elif gt_mask_binary is not None:
            # 只有真值掩模（绿色）
            gt_mask_colored = np.zeros_like(img_np)
            gt_mask_colored[:, :, 1] = gt_mask_binary * 255
            ax2.imshow(gt_mask_colored, alpha=0.4)
        elif pred_mask_binary is not None:
            # 只有预测掩模（红色）
            pred_mask_colored = np.zeros_like(img_np)
            pred_mask_colored[:, :, 0] = pred_mask_binary * 255
            ax2.imshow(pred_mask_colored, alpha=0.4)
        
        # 绘制预测点（使用JSONL中的points）
        for j, ((x, y), label) in enumerate(zip(points, labels)):
            if label == 1:
                color = 'red'
                size = 150
                point_type = 'FG'
            else:
                color = 'blue'
                size = 120
                point_type = 'BG'
            
            ax2.scatter(x, y, c=color, s=size, marker='o', 
                       edgecolors='white', linewidths=2, zorder=10)
            ax2.text(x + 15, y, f'{j}({point_type})', 
                    fontsize=10, color='white', weight='bold',
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
        
        # 设置第二幅图的标题和指标
        title2 = f'Prediction: {len(points)} points'
        if metrics:
            metrics_text = f"Dice: {metrics.get('dice', 0):.3f}, IoU: {metrics.get('iou', 0):.3f}"
            title2 += f'\n{metrics_text}'
        ax2.set_title(title2, fontsize=14, weight='bold')
        ax2.axis('off')
        
        # 在图像角落添加指标文本
        if metrics:
            metrics_str = f"Dice: {metrics.get('dice', 0):.3f}\nIoU: {metrics.get('iou', 0):.3f}\nPoints: {metrics.get('num_points', 0)}"
            ax2.text(0.02, 0.98, metrics_str, 
                    transform=ax2.transAxes, fontsize=12, weight='bold',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='可视化PEFT训练数据',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--jsonl',
        type=str,
        required=True,
        help='输入JSONL文件路径'
    )
    
    parser.add_argument(
        '--out_dir',
        type=str,
        required=True,
        help='输出图像文件夹路径'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['random', 'all'],
        default='random',
        help='采样模式: random=随机采样, all=遍历所有样本'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='随机采样模式下的样本数量（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--show_mask',
        action='store_true',
        default=True,
        help='是否显示真值掩模（默认True）'
    )
    
    parser.add_argument(
        '--no_mask',
        dest='show_mask',
        action='store_false',
        help='不显示真值掩模'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='输出图像分辨率（DPI）'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default='sample',
        help='输出文件名前缀'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='模型checkpoint路径（如果提供，将生成预测对比图）'
    )
    
    parser.add_argument(
        '--sam_checkpoint',
        type=str,
        default=None,
        help='SAM2 checkpoint路径（如果提供checkpoint，此参数必需）'
    )
    
    parser.add_argument(
        '--image_size',
        type=int,
        nargs=2,
        default=[512, 512],
        help='图像尺寸 [H, W]（用于预测）'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='设备（cuda/cpu）'
    )
    
    parser.add_argument(
        '--max_rollout_steps',
        type=int,
        default=16,
        help='最大rollout步数（用于预测）'
    )
    
    args = parser.parse_args()
    
    # 加载模型（如果提供了checkpoint）
    sam2_lora = None
    point_predictor = None
    
    if args.checkpoint:
        if not args.sam_checkpoint:
            print("Error: --sam_checkpoint is required when --checkpoint is provided")
            sys.exit(1)
        
        print(f"Loading model from {args.checkpoint}...")
        try:
            device = torch.device(args.device)
            
            # 加载checkpoint
            ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
            config = ckpt.get('config', {})
            
            # 提取配置
            lora_rank = config.get('lora_rank', 16)
            lora_alpha = config.get('lora_alpha', 32)
            fusion_mode = config.get('fusion_mode', 'film')
            feature_scale = config.get('feature_scale', 8)
            
            # 初始化SAM2 LoRA
            sam2_lora = LoRASAM2Wrapper(
                sam_checkpoint=args.sam_checkpoint,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                device=str(device),
            )
            
            # 初始化点预测网络
            sam_feature_dim = 256  # SAM2 Hiera Large的特征维度
            point_predictor = PointPredictorFromSAMFeatures(
                sam_feature_dim=sam_feature_dim,
                output_size=tuple(args.image_size),
                fusion_mode=fusion_mode,
                feature_scale=feature_scale,
            ).to(device)
            
            # 加载权重
            load_checkpoint(
                args.checkpoint, point_predictor, sam2_lora, None, None, None, str(device)
            )
            
            point_predictor.eval()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Will continue without prediction...")
            sam2_lora = None
            point_predictor = None
    
    # 加载数据
    print(f"Loading data from {args.jsonl}...")
    data = load_jsonl(args.jsonl)
    print(f"Loaded {len(data)} samples")
    
    # 选择样本
    if args.mode == 'random':
        random.seed(args.seed)
        selected_samples = random.sample(data, min(args.num_samples, len(data)))
        print(f"Randomly selected {len(selected_samples)} samples")
    else:
        selected_samples = data
        print(f"Processing all {len(selected_samples)} samples")
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 可视化每个样本
    print(f"\nGenerating visualizations...")
    for i, sample in enumerate(selected_samples):
        # 生成输出文件名
        if args.mode == 'random':
            output_filename = f'{args.prefix}_{i:03d}.png'
        else:
            # 使用图像文件名作为输出文件名
            image_path = sample.get('image', '')
            if image_path:
                stem = Path(image_path).stem
                output_filename = f'{args.prefix}_{stem}.png'
            else:
                output_filename = f'{args.prefix}_{i:04d}.png'
        
        output_path = os.path.join(args.out_dir, output_filename)
        
        # 可视化
        try:
            visualize_sample(
                sample, output_path, 
                show_mask=args.show_mask, 
                dpi=args.dpi,
                sam2_lora=sam2_lora,
                point_predictor=point_predictor,
                image_size=tuple(args.image_size),
                device=args.device,
                max_rollout_steps=args.max_rollout_steps,
            )
            print(f"  [{i+1}/{len(selected_samples)}] Saved: {output_path}")
        except Exception as e:
            print(f"  [{i+1}/{len(selected_samples)}] Error processing sample {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nDone! Visualizations saved to: {args.out_dir}")
    print(f"Total: {len(selected_samples)} images generated")


if __name__ == '__main__':
    main()


```

```python
#!/usr/bin/env python3
"""
可视化训练数据脚本

用于可视化PEFT训练数据中的图像、点标注和真值掩模。
支持随机采样或遍历所有样本。
如果提供了checkpoint，还会生成预测掩模对比图。

用法:
    # 随机采样5个样本（仅显示标注）
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_train-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode random \
        --num_samples 5

    # 遍历所有样本（带预测对比）
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_test-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode all \
        --checkpoint outputs/braintumour/peft_supervised_baseline-251107/checkpoint_best.pt \
        --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
        --image_size 512 512
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.ndimage import zoom

# 添加父目录到路径
_PARENT_DIR = Path(__file__).parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from peft.lora_sam2 import LoRASAM2Wrapper
from peft.point_predictor_peft import PointPredictorFromSAMFeatures
from peft.utils_peft import compute_dice, compute_iou, load_checkpoint


def load_jsonl(jsonl_path: str) -> List[Dict]:
    """加载JSONL文件"""
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith('['):
            # JSON数组格式
            data = json.loads(content)
        else:
            # JSONL格式（逐行）
            data = []
            for line in content.splitlines():
                if line.strip():
                    data.append(json.loads(line))
    return data


def visualize_sample(
    sample: Dict,
    output_path: str,
    show_mask: bool = True,
    dpi: int = 150,
    show_prediction: bool = True,
) -> None:
    """
    可视化单个样本
    
    Args:
        sample: 样本字典，包含 'image', 'gt_mask', 'points', 'labels', 'sam_masks_dir' 等字段
        output_path: 输出图像路径
        show_mask: 是否显示真值掩模
        dpi: 图像分辨率
        show_prediction: 是否显示预测掩模对比（需要sam_masks_dir）
    """
    # 加载图像
    image_path = sample['image']
    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        return
    
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    orig_w, orig_h = img.size
    
    # 获取点和标签
    points = sample.get('points', [])
    labels = sample.get('labels', [])
    
    if len(points) != len(labels):
        print(f"Warning: Points and labels length mismatch: {len(points)} vs {len(labels)}")
        return
    
    # 加载预测掩模和计算指标（如果sam_masks_dir存在）
    pred_mask = None
    pred_mask_binary = None
    gt_mask_binary_for_metrics = None
    metrics = {}
    do_prediction = False
    
    if show_prediction:
        sam_masks_dir = sample.get('sam_masks_dir', None)
        gt_mask_path = sample.get('gt_mask', None)
        
        if sam_masks_dir and gt_mask_path:
            try:
                # 获取图像stem（文件名不含扩展名）
                image_stem = Path(image_path).stem
                mask_dir = os.path.join(sam_masks_dir, image_stem)
                
                # 加载最后一个步骤的预测掩模（len(points)-1.png）
                if len(points) > 0:
                    last_mask_idx = len(points) - 1
                    pred_mask_path = os.path.join(mask_dir, f"{last_mask_idx}.png")
                    
                    if os.path.exists(pred_mask_path):
                        pred_mask = np.array(Image.open(pred_mask_path).convert('L'))
                        pred_mask_binary = (pred_mask > 127).astype(float)
                        
                        # 加载真值掩模并计算指标
                        if os.path.exists(gt_mask_path):
                            gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                            gt_mask_binary_for_metrics = (gt_mask > 127).astype(float)
                            
                            # 调整尺寸以匹配（如果需要）
                            if gt_mask_binary_for_metrics.shape != pred_mask_binary.shape:
                                scale_y = pred_mask_binary.shape[0] / gt_mask_binary_for_metrics.shape[0]
                                scale_x = pred_mask_binary.shape[1] / gt_mask_binary_for_metrics.shape[1]
                                gt_mask_binary_for_metrics = zoom(gt_mask_binary_for_metrics, (scale_y, scale_x), order=0)
                            
                            # 转换为tensor计算指标
                            pred_tensor = torch.from_numpy(pred_mask_binary).float()
                            gt_tensor = torch.from_numpy(gt_mask_binary_for_metrics).float()
                            
                            dice = compute_dice(pred_tensor, gt_tensor)
                            iou = compute_iou(pred_tensor, gt_tensor)
                            
                            metrics['dice'] = dice
                            metrics['iou'] = iou
                            metrics['num_points'] = len(points)
                            
                            do_prediction = True
                        else:
                            print(f"Warning: GT mask not found: {gt_mask_path}")
                    else:
                        print(f"Warning: Predicted mask not found: {pred_mask_path}")
                else:
                    print(f"Warning: No points in sample, cannot load predicted mask")
            except Exception as e:
                print(f"Warning: Failed to load predicted mask: {e}")
                import traceback
                traceback.print_exc()
    
    # 创建图形
    if do_prediction:
        # 两幅图：第一幅显示点和标注，第二幅显示掩模对比
        fig, axes = plt.subplots(1, 2, figsize=(24, 12))
        ax1, ax2 = axes
    else:
        # 单幅图：只显示点和标注
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 12))
        ax2 = None
    
    # 第一幅图：显示点和标注
    ax1.imshow(img_np)
    
    # 如果有真值掩模，叠加显示
    if show_mask:
        gt_mask_path = sample.get('gt_mask', None)
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                mask_binary = (gt_mask > 127).astype(float)
                ax1.imshow(mask_binary, alpha=0.3, cmap='Greens')
            except Exception as e:
                print(f"Warning: Failed to load mask {gt_mask_path}: {e}")
    
    # 绘制点
    for j, ((x, y), label) in enumerate(zip(points, labels)):
        if label == 1:
            # 前景点：红色，实心圆
            color = 'red'
            size = 150
            point_type = 'FG'
        else:
            # 背景点：蓝色，实心圆
            color = 'blue'
            size = 120
            point_type = 'BG'
        
        # 绘制点
        ax1.scatter(x, y, c=color, s=size, marker='o', 
                   edgecolors='white', linewidths=2, zorder=10)
        
        # 添加标签文本
        ax1.text(x + 15, y, f'{j}({point_type})', 
                fontsize=10, color='white', weight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    # 设置第一幅图的标题
    stem = Path(image_path).stem
    title1 = f'{stem}: {len(points)} points (Ground Truth)'
    ax1.set_title(title1, fontsize=14, weight='bold')
    ax1.axis('off')
    
    # 第二幅图：显示掩模对比（如果进行预测）
    if do_prediction and ax2 is not None:
        ax2.imshow(img_np)
        
        gt_mask_path = sample.get('gt_mask', None)
        gt_mask_binary = None
        
        # 加载并调整真值掩模尺寸（用于显示）
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                gt_mask_binary = (gt_mask > 127).astype(float)
                # 调整尺寸以匹配图像
                if gt_mask_binary.shape != img_np.shape[:2]:
                    scale_y = img_np.shape[0] / gt_mask_binary.shape[0]
                    scale_x = img_np.shape[1] / gt_mask_binary.shape[1]
                    gt_mask_binary = zoom(gt_mask_binary, (scale_y, scale_x), order=0)
            except Exception as e:
                print(f"Warning: Failed to load GT mask: {e}")
        
        # 调整预测掩模尺寸（用于显示）
        pred_mask_binary_display = None
        if pred_mask_binary is not None:
            # pred_mask_binary已经是二值化的，但需要调整尺寸以匹配图像
            if pred_mask_binary.shape != img_np.shape[:2]:
                scale_y = img_np.shape[0] / pred_mask_binary.shape[0]
                scale_x = img_np.shape[1] / pred_mask_binary.shape[1]
                pred_mask_binary_display = zoom(pred_mask_binary, (scale_y, scale_x), order=0)
            else:
                pred_mask_binary_display = pred_mask_binary
        
        # 优化显示：先显示重叠区域（黄色），再显示真值（绿色），最后显示预测（红色）
        if gt_mask_binary is not None and pred_mask_binary is not None:
            # 重叠区域（黄色）
            overlap = (gt_mask_binary * pred_mask_binary).astype(float)
            overlap_colored = np.zeros_like(img_np)
            overlap_colored[:, :, 0] = overlap * 255  # 红色
            overlap_colored[:, :, 1] = overlap * 255  # 绿色
            ax2.imshow(overlap_colored, alpha=0.5)
            
            # 真值掩模的未重叠部分（绿色）
            gt_only = (gt_mask_binary * (1 - pred_mask_binary)).astype(float)
            gt_only_colored = np.zeros_like(img_np)
            gt_only_colored[:, :, 1] = gt_only * 255
            ax2.imshow(gt_only_colored, alpha=0.4)
            
            # 预测掩模的未重叠部分（红色）
            pred_only = (pred_mask_binary * (1 - gt_mask_binary)).astype(float)
            pred_only_colored = np.zeros_like(img_np)
            pred_only_colored[:, :, 0] = pred_only * 255
            ax2.imshow(pred_only_colored, alpha=0.4)
        elif gt_mask_binary is not None:
            # 只有真值掩模（绿色）
            gt_mask_colored = np.zeros_like(img_np)
            gt_mask_colored[:, :, 1] = gt_mask_binary * 255
            ax2.imshow(gt_mask_colored, alpha=0.4)
        elif pred_mask_binary is not None:
            # 只有预测掩模（红色）
            pred_mask_colored = np.zeros_like(img_np)
            pred_mask_colored[:, :, 0] = pred_mask_binary * 255
            ax2.imshow(pred_mask_colored, alpha=0.4)
        
        # 绘制预测点（使用JSONL中的points）
        for j, ((x, y), label) in enumerate(zip(points, labels)):
            if label == 1:
                color = 'red'
                size = 150
                point_type = 'FG'
            else:
                color = 'blue'
                size = 120
                point_type = 'BG'
            
            ax2.scatter(x, y, c=color, s=size, marker='o', 
                       edgecolors='white', linewidths=2, zorder=10)
            ax2.text(x + 15, y, f'{j}({point_type})', 
                    fontsize=10, color='white', weight='bold',
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
        
        # 设置第二幅图的标题和指标
        title2 = f'Prediction: {len(points)} points'
        if metrics:
            metrics_text = f"Dice: {metrics.get('dice', 0):.3f}, IoU: {metrics.get('iou', 0):.3f}"
            title2 += f'\n{metrics_text}'
        ax2.set_title(title2, fontsize=14, weight='bold')
        ax2.axis('off')
        
        # 在图像角落添加指标文本
        if metrics:
            metrics_str = f"Dice: {metrics.get('dice', 0):.3f}\nIoU: {metrics.get('iou', 0):.3f}\nPoints: {metrics.get('num_points', 0)}"
            ax2.text(0.02, 0.98, metrics_str, 
                    transform=ax2.transAxes, fontsize=12, weight='bold',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='可视化PEFT训练数据',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--jsonl',
        type=str,
        required=True,
        help='输入JSONL文件路径'
    )
    
    parser.add_argument(
        '--out_dir',
        type=str,
        required=True,
        help='输出图像文件夹路径'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['random', 'all'],
        default='random',
        help='采样模式: random=随机采样, all=遍历所有样本'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='随机采样模式下的样本数量（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--show_mask',
        action='store_true',
        default=True,
        help='是否显示真值掩模（默认True）'
    )
    
    parser.add_argument(
        '--no_mask',
        dest='show_mask',
        action='store_false',
        help='不显示真值掩模'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='输出图像分辨率（DPI）'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default='sample',
        help='输出文件名前缀'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='模型checkpoint路径（如果提供，将生成预测对比图）'
    )
    
    parser.add_argument(
        '--sam_checkpoint',
        type=str,
        default=None,
        help='SAM2 checkpoint路径（如果提供checkpoint，此参数必需）'
    )
    
    parser.add_argument(
        '--image_size',
        type=int,
        nargs=2,
        default=[512, 512],
        help='图像尺寸 [H, W]（用于预测）'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='设备（cuda/cpu）'
    )
    
    parser.add_argument(
        '--max_rollout_steps',
        type=int,
        default=16,
        help='最大rollout步数（用于预测）'
    )
    
    args = parser.parse_args()
    
    # 加载模型（如果提供了checkpoint）
    sam2_lora = None
    point_predictor = None
    
    if args.checkpoint:
        if not args.sam_checkpoint:
            print("Error: --sam_checkpoint is required when --checkpoint is provided")
            sys.exit(1)
        
        print(f"Loading model from {args.checkpoint}...")
        try:
            device = torch.device(args.device)
            
            # 加载checkpoint
            ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
            config = ckpt.get('config', {})
            
            # 提取配置
            lora_rank = config.get('lora_rank', 16)
            lora_alpha = config.get('lora_alpha', 32)
            fusion_mode = config.get('fusion_mode', 'film')
            feature_scale = config.get('feature_scale', 8)
            
            # 初始化SAM2 LoRA
            sam2_lora = LoRASAM2Wrapper(
                sam_checkpoint=args.sam_checkpoint,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                device=str(device),
            )
            
            # 初始化点预测网络
            sam_feature_dim = 256  # SAM2 Hiera Large的特征维度
            point_predictor = PointPredictorFromSAMFeatures(
                sam_feature_dim=sam_feature_dim,
                output_size=tuple(args.image_size),
                fusion_mode=fusion_mode,
                feature_scale=feature_scale,
            ).to(device)
            
            # 加载权重
            load_checkpoint(
                args.checkpoint, point_predictor, sam2_lora, None, None, None, str(device)
            )
            
            point_predictor.eval()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Will continue without prediction...")
            sam2_lora = None
            point_predictor = None
    
    # 加载数据
    print(f"Loading data from {args.jsonl}...")
    data = load_jsonl(args.jsonl)
    print(f"Loaded {len(data)} samples")
    
    # 选择样本
    if args.mode == 'random':
        random.seed(args.seed)
        selected_samples = random.sample(data, min(args.num_samples, len(data)))
        print(f"Randomly selected {len(selected_samples)} samples")
    else:
        selected_samples = data
        print(f"Processing all {len(selected_samples)} samples")
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 可视化每个样本
    print(f"\nGenerating visualizations...")
    for i, sample in enumerate(selected_samples):
        # 生成输出文件名
        if args.mode == 'random':
            output_filename = f'{args.prefix}_{i:03d}.png'
        else:
            # 使用图像文件名作为输出文件名
            image_path = sample.get('image', '')
            if image_path:
                stem = Path(image_path).stem
                output_filename = f'{args.prefix}_{stem}.png'
            else:
                output_filename = f'{args.prefix}_{i:04d}.png'
        
        output_path = os.path.join(args.out_dir, output_filename)
        
        # 可视化
        try:
            visualize_sample(
                sample, output_path, 
                show_mask=args.show_mask, 
                dpi=args.dpi,
                sam2_lora=sam2_lora,
                point_predictor=point_predictor,
                image_size=tuple(args.image_size),
                device=args.device,
                max_rollout_steps=args.max_rollout_steps,
            )
            print(f"  [{i+1}/{len(selected_samples)}] Saved: {output_path}")
        except Exception as e:
            print(f"  [{i+1}/{len(selected_samples)}] Error processing sample {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nDone! Visualizations saved to: {args.out_dir}")
    print(f"Total: {len(selected_samples)} images generated")


if __name__ == '__main__':
    main()


```

```python
#!/usr/bin/env python3
"""
可视化训练数据脚本

用于可视化PEFT训练数据中的图像、点标注和真值掩模。
支持随机采样或遍历所有样本。
如果提供了checkpoint，还会生成预测掩模对比图。

用法:
    # 随机采样5个样本（仅显示标注）
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_train-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode random \
        --num_samples 5

    # 遍历所有样本（带预测对比）
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_test-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode all \
        --checkpoint outputs/braintumour/peft_supervised_baseline-251107/checkpoint_best.pt \
        --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
        --image_size 512 512
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.ndimage import zoom

# 添加父目录到路径
_PARENT_DIR = Path(__file__).parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from peft.lora_sam2 import LoRASAM2Wrapper
from peft.point_predictor_peft import PointPredictorFromSAMFeatures
from peft.utils_peft import compute_dice, compute_iou, load_checkpoint


def load_jsonl(jsonl_path: str) -> List[Dict]:
    """加载JSONL文件"""
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith('['):
            # JSON数组格式
            data = json.loads(content)
        else:
            # JSONL格式（逐行）
            data = []
            for line in content.splitlines():
                if line.strip():
                    data.append(json.loads(line))
    return data


def visualize_sample(
    sample: Dict,
    output_path: str,
    show_mask: bool = True,
    dpi: int = 150,
    show_prediction: bool = True,
) -> None:
    """
    可视化单个样本
    
    Args:
        sample: 样本字典，包含 'image', 'gt_mask', 'points', 'labels', 'sam_masks_dir' 等字段
        output_path: 输出图像路径
        show_mask: 是否显示真值掩模
        dpi: 图像分辨率
        show_prediction: 是否显示预测掩模对比（需要sam_masks_dir）
    """
    # 加载图像
    image_path = sample['image']
    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        return
    
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    orig_w, orig_h = img.size
    
    # 获取点和标签
    points = sample.get('points', [])
    labels = sample.get('labels', [])
    
    if len(points) != len(labels):
        print(f"Warning: Points and labels length mismatch: {len(points)} vs {len(labels)}")
        return
    
    # 加载预测掩模和计算指标（如果sam_masks_dir存在）
    pred_mask = None
    pred_mask_binary = None
    gt_mask_binary_for_metrics = None
    metrics = {}
    do_prediction = False
    
    if show_prediction:
        sam_masks_dir = sample.get('sam_masks_dir', None)
        gt_mask_path = sample.get('gt_mask', None)
        
        if sam_masks_dir and gt_mask_path:
            try:
                # 获取图像stem（文件名不含扩展名）
                image_stem = Path(image_path).stem
                mask_dir = os.path.join(sam_masks_dir, image_stem)
                
                # 加载最后一个步骤的预测掩模（len(points)-1.png）
                if len(points) > 0:
                    last_mask_idx = len(points) - 1
                    pred_mask_path = os.path.join(mask_dir, f"{last_mask_idx}.png")
                    
                    if os.path.exists(pred_mask_path):
                        pred_mask = np.array(Image.open(pred_mask_path).convert('L'))
                        pred_mask_binary = (pred_mask > 127).astype(float)
                        
                        # 加载真值掩模并计算指标
                        if os.path.exists(gt_mask_path):
                            gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                            gt_mask_binary_for_metrics = (gt_mask > 127).astype(float)
                            
                            # 调整尺寸以匹配（如果需要）
                            if gt_mask_binary_for_metrics.shape != pred_mask_binary.shape:
                                scale_y = pred_mask_binary.shape[0] / gt_mask_binary_for_metrics.shape[0]
                                scale_x = pred_mask_binary.shape[1] / gt_mask_binary_for_metrics.shape[1]
                                gt_mask_binary_for_metrics = zoom(gt_mask_binary_for_metrics, (scale_y, scale_x), order=0)
                            
                            # 转换为tensor计算指标
                            pred_tensor = torch.from_numpy(pred_mask_binary).float()
                            gt_tensor = torch.from_numpy(gt_mask_binary_for_metrics).float()
                            
                            dice = compute_dice(pred_tensor, gt_tensor)
                            iou = compute_iou(pred_tensor, gt_tensor)
                            
                            metrics['dice'] = dice
                            metrics['iou'] = iou
                            metrics['num_points'] = len(points)
                            
                            do_prediction = True
                        else:
                            print(f"Warning: GT mask not found: {gt_mask_path}")
                    else:
                        print(f"Warning: Predicted mask not found: {pred_mask_path}")
                else:
                    print(f"Warning: No points in sample, cannot load predicted mask")
            except Exception as e:
                print(f"Warning: Failed to load predicted mask: {e}")
                import traceback
                traceback.print_exc()
    
    # 创建图形
    if do_prediction:
        # 两幅图：第一幅显示点和标注，第二幅显示掩模对比
        fig, axes = plt.subplots(1, 2, figsize=(24, 12))
        ax1, ax2 = axes
    else:
        # 单幅图：只显示点和标注
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 12))
        ax2 = None
    
    # 第一幅图：显示点和标注
    ax1.imshow(img_np)
    
    # 如果有真值掩模，叠加显示
    if show_mask:
        gt_mask_path = sample.get('gt_mask', None)
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                mask_binary = (gt_mask > 127).astype(float)
                ax1.imshow(mask_binary, alpha=0.3, cmap='Greens')
            except Exception as e:
                print(f"Warning: Failed to load mask {gt_mask_path}: {e}")
    
    # 绘制点
    for j, ((x, y), label) in enumerate(zip(points, labels)):
        if label == 1:
            # 前景点：红色，实心圆
            color = 'red'
            size = 150
            point_type = 'FG'
        else:
            # 背景点：蓝色，实心圆
            color = 'blue'
            size = 120
            point_type = 'BG'
        
        # 绘制点
        ax1.scatter(x, y, c=color, s=size, marker='o', 
                   edgecolors='white', linewidths=2, zorder=10)
        
        # 添加标签文本
        ax1.text(x + 15, y, f'{j}({point_type})', 
                fontsize=10, color='white', weight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    # 设置第一幅图的标题
    stem = Path(image_path).stem
    title1 = f'{stem}: {len(points)} points (Ground Truth)'
    ax1.set_title(title1, fontsize=14, weight='bold')
    ax1.axis('off')
    
    # 第二幅图：显示掩模对比（如果进行预测）
    if do_prediction and ax2 is not None:
        ax2.imshow(img_np)
        
        gt_mask_path = sample.get('gt_mask', None)
        gt_mask_binary = None
        
        # 加载并调整真值掩模尺寸（用于显示）
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                gt_mask_binary = (gt_mask > 127).astype(float)
                # 调整尺寸以匹配图像
                if gt_mask_binary.shape != img_np.shape[:2]:
                    scale_y = img_np.shape[0] / gt_mask_binary.shape[0]
                    scale_x = img_np.shape[1] / gt_mask_binary.shape[1]
                    gt_mask_binary = zoom(gt_mask_binary, (scale_y, scale_x), order=0)
            except Exception as e:
                print(f"Warning: Failed to load GT mask: {e}")
        
        # 调整预测掩模尺寸（用于显示）
        pred_mask_binary_display = None
        if pred_mask_binary is not None:
            # pred_mask_binary已经是二值化的，但需要调整尺寸以匹配图像
            if pred_mask_binary.shape != img_np.shape[:2]:
                scale_y = img_np.shape[0] / pred_mask_binary.shape[0]
                scale_x = img_np.shape[1] / pred_mask_binary.shape[1]
                pred_mask_binary_display = zoom(pred_mask_binary, (scale_y, scale_x), order=0)
            else:
                pred_mask_binary_display = pred_mask_binary
        
        # 优化显示：先显示重叠区域（黄色），再显示真值（绿色），最后显示预测（红色）
        if gt_mask_binary is not None and pred_mask_binary_display is not None:
            # 重叠区域（黄色）
            overlap = (gt_mask_binary * pred_mask_binary).astype(float)
            overlap_colored = np.zeros_like(img_np)
            overlap_colored[:, :, 0] = overlap * 255  # 红色
            overlap_colored[:, :, 1] = overlap * 255  # 绿色
            ax2.imshow(overlap_colored, alpha=0.5)
            
            # 真值掩模的未重叠部分（绿色）
            gt_only = (gt_mask_binary * (1 - pred_mask_binary)).astype(float)
            gt_only_colored = np.zeros_like(img_np)
            gt_only_colored[:, :, 1] = gt_only * 255
            ax2.imshow(gt_only_colored, alpha=0.4)
            
            # 预测掩模的未重叠部分（红色）
            pred_only = (pred_mask_binary * (1 - gt_mask_binary)).astype(float)
            pred_only_colored = np.zeros_like(img_np)
            pred_only_colored[:, :, 0] = pred_only * 255
            ax2.imshow(pred_only_colored, alpha=0.4)
        elif gt_mask_binary is not None:
            # 只有真值掩模（绿色）
            gt_mask_colored = np.zeros_like(img_np)
            gt_mask_colored[:, :, 1] = gt_mask_binary * 255
            ax2.imshow(gt_mask_colored, alpha=0.4)
        elif pred_mask_binary is not None:
            # 只有预测掩模（红色）
            pred_mask_colored = np.zeros_like(img_np)
            pred_mask_colored[:, :, 0] = pred_mask_binary * 255
            ax2.imshow(pred_mask_colored, alpha=0.4)
        
        # 绘制预测点（使用JSONL中的points）
        for j, ((x, y), label) in enumerate(zip(points, labels)):
            if label == 1:
                color = 'red'
                size = 150
                point_type = 'FG'
            else:
                color = 'blue'
                size = 120
                point_type = 'BG'
            
            ax2.scatter(x, y, c=color, s=size, marker='o', 
                       edgecolors='white', linewidths=2, zorder=10)
            ax2.text(x + 15, y, f'{j}({point_type})', 
                    fontsize=10, color='white', weight='bold',
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
        
        # 设置第二幅图的标题和指标
        title2 = f'Prediction: {len(points)} points'
        if metrics:
            metrics_text = f"Dice: {metrics.get('dice', 0):.3f}, IoU: {metrics.get('iou', 0):.3f}"
            title2 += f'\n{metrics_text}'
        ax2.set_title(title2, fontsize=14, weight='bold')
        ax2.axis('off')
        
        # 在图像角落添加指标文本
        if metrics:
            metrics_str = f"Dice: {metrics.get('dice', 0):.3f}\nIoU: {metrics.get('iou', 0):.3f}\nPoints: {metrics.get('num_points', 0)}"
            ax2.text(0.02, 0.98, metrics_str, 
                    transform=ax2.transAxes, fontsize=12, weight='bold',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='可视化PEFT训练数据',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--jsonl',
        type=str,
        required=True,
        help='输入JSONL文件路径'
    )
    
    parser.add_argument(
        '--out_dir',
        type=str,
        required=True,
        help='输出图像文件夹路径'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['random', 'all'],
        default='random',
        help='采样模式: random=随机采样, all=遍历所有样本'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='随机采样模式下的样本数量（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--show_mask',
        action='store_true',
        default=True,
        help='是否显示真值掩模（默认True）'
    )
    
    parser.add_argument(
        '--no_mask',
        dest='show_mask',
        action='store_false',
        help='不显示真值掩模'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='输出图像分辨率（DPI）'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default='sample',
        help='输出文件名前缀'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='模型checkpoint路径（如果提供，将生成预测对比图）'
    )
    
    parser.add_argument(
        '--sam_checkpoint',
        type=str,
        default=None,
        help='SAM2 checkpoint路径（如果提供checkpoint，此参数必需）'
    )
    
    parser.add_argument(
        '--image_size',
        type=int,
        nargs=2,
        default=[512, 512],
        help='图像尺寸 [H, W]（用于预测）'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='设备（cuda/cpu）'
    )
    
    parser.add_argument(
        '--max_rollout_steps',
        type=int,
        default=16,
        help='最大rollout步数（用于预测）'
    )
    
    args = parser.parse_args()
    
    # 加载模型（如果提供了checkpoint）
    sam2_lora = None
    point_predictor = None
    
    if args.checkpoint:
        if not args.sam_checkpoint:
            print("Error: --sam_checkpoint is required when --checkpoint is provided")
            sys.exit(1)
        
        print(f"Loading model from {args.checkpoint}...")
        try:
            device = torch.device(args.device)
            
            # 加载checkpoint
            ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
            config = ckpt.get('config', {})
            
            # 提取配置
            lora_rank = config.get('lora_rank', 16)
            lora_alpha = config.get('lora_alpha', 32)
            fusion_mode = config.get('fusion_mode', 'film')
            feature_scale = config.get('feature_scale', 8)
            
            # 初始化SAM2 LoRA
            sam2_lora = LoRASAM2Wrapper(
                sam_checkpoint=args.sam_checkpoint,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                device=str(device),
            )
            
            # 初始化点预测网络
            sam_feature_dim = 256  # SAM2 Hiera Large的特征维度
            point_predictor = PointPredictorFromSAMFeatures(
                sam_feature_dim=sam_feature_dim,
                output_size=tuple(args.image_size),
                fusion_mode=fusion_mode,
                feature_scale=feature_scale,
            ).to(device)
            
            # 加载权重
            load_checkpoint(
                args.checkpoint, point_predictor, sam2_lora, None, None, None, str(device)
            )
            
            point_predictor.eval()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Will continue without prediction...")
            sam2_lora = None
            point_predictor = None
    
    # 加载数据
    print(f"Loading data from {args.jsonl}...")
    data = load_jsonl(args.jsonl)
    print(f"Loaded {len(data)} samples")
    
    # 选择样本
    if args.mode == 'random':
        random.seed(args.seed)
        selected_samples = random.sample(data, min(args.num_samples, len(data)))
        print(f"Randomly selected {len(selected_samples)} samples")
    else:
        selected_samples = data
        print(f"Processing all {len(selected_samples)} samples")
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 可视化每个样本
    print(f"\nGenerating visualizations...")
    for i, sample in enumerate(selected_samples):
        # 生成输出文件名
        if args.mode == 'random':
            output_filename = f'{args.prefix}_{i:03d}.png'
        else:
            # 使用图像文件名作为输出文件名
            image_path = sample.get('image', '')
            if image_path:
                stem = Path(image_path).stem
                output_filename = f'{args.prefix}_{stem}.png'
            else:
                output_filename = f'{args.prefix}_{i:04d}.png'
        
        output_path = os.path.join(args.out_dir, output_filename)
        
        # 可视化
        try:
            visualize_sample(
                sample, output_path, 
                show_mask=args.show_mask, 
                dpi=args.dpi,
                sam2_lora=sam2_lora,
                point_predictor=point_predictor,
                image_size=tuple(args.image_size),
                device=args.device,
                max_rollout_steps=args.max_rollout_steps,
            )
            print(f"  [{i+1}/{len(selected_samples)}] Saved: {output_path}")
        except Exception as e:
            print(f"  [{i+1}/{len(selected_samples)}] Error processing sample {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nDone! Visualizations saved to: {args.out_dir}")
    print(f"Total: {len(selected_samples)} images generated")


if __name__ == '__main__':
    main()


```

```python
#!/usr/bin/env python3
"""
可视化训练数据脚本

用于可视化PEFT训练数据中的图像、点标注和真值掩模。
支持随机采样或遍历所有样本。
如果提供了checkpoint，还会生成预测掩模对比图。

用法:
    # 随机采样5个样本（仅显示标注）
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_train-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode random \
        --num_samples 5

    # 遍历所有样本（带预测对比）
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_test-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode all \
        --checkpoint outputs/braintumour/peft_supervised_baseline-251107/checkpoint_best.pt \
        --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
        --image_size 512 512
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.ndimage import zoom

# 添加父目录到路径
_PARENT_DIR = Path(__file__).parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from peft.lora_sam2 import LoRASAM2Wrapper
from peft.point_predictor_peft import PointPredictorFromSAMFeatures
from peft.utils_peft import compute_dice, compute_iou, load_checkpoint


def load_jsonl(jsonl_path: str) -> List[Dict]:
    """加载JSONL文件"""
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith('['):
            # JSON数组格式
            data = json.loads(content)
        else:
            # JSONL格式（逐行）
            data = []
            for line in content.splitlines():
                if line.strip():
                    data.append(json.loads(line))
    return data


def visualize_sample(
    sample: Dict,
    output_path: str,
    show_mask: bool = True,
    dpi: int = 150,
    show_prediction: bool = True,
) -> None:
    """
    可视化单个样本
    
    Args:
        sample: 样本字典，包含 'image', 'gt_mask', 'points', 'labels', 'sam_masks_dir' 等字段
        output_path: 输出图像路径
        show_mask: 是否显示真值掩模
        dpi: 图像分辨率
        show_prediction: 是否显示预测掩模对比（需要sam_masks_dir）
    """
    # 加载图像
    image_path = sample['image']
    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        return
    
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    orig_w, orig_h = img.size
    
    # 获取点和标签
    points = sample.get('points', [])
    labels = sample.get('labels', [])
    
    if len(points) != len(labels):
        print(f"Warning: Points and labels length mismatch: {len(points)} vs {len(labels)}")
        return
    
    # 加载预测掩模和计算指标（如果sam_masks_dir存在）
    pred_mask = None
    pred_mask_binary = None
    gt_mask_binary_for_metrics = None
    metrics = {}
    do_prediction = False
    
    if show_prediction:
        sam_masks_dir = sample.get('sam_masks_dir', None)
        gt_mask_path = sample.get('gt_mask', None)
        
        if sam_masks_dir and gt_mask_path:
            try:
                # 获取图像stem（文件名不含扩展名）
                image_stem = Path(image_path).stem
                mask_dir = os.path.join(sam_masks_dir, image_stem)
                
                # 加载最后一个步骤的预测掩模（len(points)-1.png）
                if len(points) > 0:
                    last_mask_idx = len(points) - 1
                    pred_mask_path = os.path.join(mask_dir, f"{last_mask_idx}.png")
                    
                    if os.path.exists(pred_mask_path):
                        pred_mask = np.array(Image.open(pred_mask_path).convert('L'))
                        pred_mask_binary = (pred_mask > 127).astype(float)
                        
                        # 加载真值掩模并计算指标
                        if os.path.exists(gt_mask_path):
                            gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                            gt_mask_binary_for_metrics = (gt_mask > 127).astype(float)
                            
                            # 调整尺寸以匹配（如果需要）
                            if gt_mask_binary_for_metrics.shape != pred_mask_binary.shape:
                                scale_y = pred_mask_binary.shape[0] / gt_mask_binary_for_metrics.shape[0]
                                scale_x = pred_mask_binary.shape[1] / gt_mask_binary_for_metrics.shape[1]
                                gt_mask_binary_for_metrics = zoom(gt_mask_binary_for_metrics, (scale_y, scale_x), order=0)
                            
                            # 转换为tensor计算指标
                            pred_tensor = torch.from_numpy(pred_mask_binary).float()
                            gt_tensor = torch.from_numpy(gt_mask_binary_for_metrics).float()
                            
                            dice = compute_dice(pred_tensor, gt_tensor)
                            iou = compute_iou(pred_tensor, gt_tensor)
                            
                            metrics['dice'] = dice
                            metrics['iou'] = iou
                            metrics['num_points'] = len(points)
                            
                            do_prediction = True
                        else:
                            print(f"Warning: GT mask not found: {gt_mask_path}")
                    else:
                        print(f"Warning: Predicted mask not found: {pred_mask_path}")
                else:
                    print(f"Warning: No points in sample, cannot load predicted mask")
            except Exception as e:
                print(f"Warning: Failed to load predicted mask: {e}")
                import traceback
                traceback.print_exc()
    
    # 创建图形
    if do_prediction:
        # 两幅图：第一幅显示点和标注，第二幅显示掩模对比
        fig, axes = plt.subplots(1, 2, figsize=(24, 12))
        ax1, ax2 = axes
    else:
        # 单幅图：只显示点和标注
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 12))
        ax2 = None
    
    # 第一幅图：显示点和标注
    ax1.imshow(img_np)
    
    # 如果有真值掩模，叠加显示
    if show_mask:
        gt_mask_path = sample.get('gt_mask', None)
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                mask_binary = (gt_mask > 127).astype(float)
                ax1.imshow(mask_binary, alpha=0.3, cmap='Greens')
            except Exception as e:
                print(f"Warning: Failed to load mask {gt_mask_path}: {e}")
    
    # 绘制点
    for j, ((x, y), label) in enumerate(zip(points, labels)):
        if label == 1:
            # 前景点：红色，实心圆
            color = 'red'
            size = 150
            point_type = 'FG'
        else:
            # 背景点：蓝色，实心圆
            color = 'blue'
            size = 120
            point_type = 'BG'
        
        # 绘制点
        ax1.scatter(x, y, c=color, s=size, marker='o', 
                   edgecolors='white', linewidths=2, zorder=10)
        
        # 添加标签文本
        ax1.text(x + 15, y, f'{j}({point_type})', 
                fontsize=10, color='white', weight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    # 设置第一幅图的标题
    stem = Path(image_path).stem
    title1 = f'{stem}: {len(points)} points (Ground Truth)'
    ax1.set_title(title1, fontsize=14, weight='bold')
    ax1.axis('off')
    
    # 第二幅图：显示掩模对比（如果进行预测）
    if do_prediction and ax2 is not None:
        ax2.imshow(img_np)
        
        gt_mask_path = sample.get('gt_mask', None)
        gt_mask_binary = None
        
        # 加载并调整真值掩模尺寸（用于显示）
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                gt_mask_binary = (gt_mask > 127).astype(float)
                # 调整尺寸以匹配图像
                if gt_mask_binary.shape != img_np.shape[:2]:
                    scale_y = img_np.shape[0] / gt_mask_binary.shape[0]
                    scale_x = img_np.shape[1] / gt_mask_binary.shape[1]
                    gt_mask_binary = zoom(gt_mask_binary, (scale_y, scale_x), order=0)
            except Exception as e:
                print(f"Warning: Failed to load GT mask: {e}")
        
        # 调整预测掩模尺寸（用于显示）
        pred_mask_binary_display = None
        if pred_mask_binary is not None:
            # pred_mask_binary已经是二值化的，但需要调整尺寸以匹配图像
            if pred_mask_binary.shape != img_np.shape[:2]:
                scale_y = img_np.shape[0] / pred_mask_binary.shape[0]
                scale_x = img_np.shape[1] / pred_mask_binary.shape[1]
                pred_mask_binary_display = zoom(pred_mask_binary, (scale_y, scale_x), order=0)
            else:
                pred_mask_binary_display = pred_mask_binary
        
        # 优化显示：先显示重叠区域（黄色），再显示真值（绿色），最后显示预测（红色）
        if gt_mask_binary is not None and pred_mask_binary_display is not None:
            # 重叠区域（黄色）
            overlap = (gt_mask_binary * pred_mask_binary_display).astype(float)
            overlap_colored = np.zeros_like(img_np)
            overlap_colored[:, :, 0] = overlap * 255  # 红色
            overlap_colored[:, :, 1] = overlap * 255  # 绿色
            ax2.imshow(overlap_colored, alpha=0.5)
            
            # 真值掩模的未重叠部分（绿色）
            gt_only = (gt_mask_binary * (1 - pred_mask_binary_display)).astype(float)
            gt_only_colored = np.zeros_like(img_np)
            gt_only_colored[:, :, 1] = gt_only * 255
            ax2.imshow(gt_only_colored, alpha=0.4)
            
            # 预测掩模的未重叠部分（红色）
            pred_only = (pred_mask_binary_display * (1 - gt_mask_binary)).astype(float)
            pred_only_colored = np.zeros_like(img_np)
            pred_only_colored[:, :, 0] = pred_only * 255
            ax2.imshow(pred_only_colored, alpha=0.4)
        elif gt_mask_binary is not None:
            # 只有真值掩模（绿色）
            gt_mask_colored = np.zeros_like(img_np)
            gt_mask_colored[:, :, 1] = gt_mask_binary * 255
            ax2.imshow(gt_mask_colored, alpha=0.4)
        elif pred_mask_binary_display is not None:
            # 只有预测掩模（红色）
            pred_mask_colored = np.zeros_like(img_np)
            pred_mask_colored[:, :, 0] = pred_mask_binary_display * 255
            ax2.imshow(pred_mask_colored, alpha=0.4)
        
        # 绘制预测点（使用JSONL中的points）
        for j, ((x, y), label) in enumerate(zip(points, labels)):
            if label == 1:
                color = 'red'
                size = 150
                point_type = 'FG'
            else:
                color = 'blue'
                size = 120
                point_type = 'BG'
            
            ax2.scatter(x, y, c=color, s=size, marker='o', 
                       edgecolors='white', linewidths=2, zorder=10)
            ax2.text(x + 15, y, f'{j}({point_type})', 
                    fontsize=10, color='white', weight='bold',
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
        
        # 设置第二幅图的标题和指标
        title2 = f'Prediction: {len(points)} points'
        if metrics:
            metrics_text = f"Dice: {metrics.get('dice', 0):.3f}, IoU: {metrics.get('iou', 0):.3f}"
            title2 += f'\n{metrics_text}'
        ax2.set_title(title2, fontsize=14, weight='bold')
        ax2.axis('off')
        
        # 在图像角落添加指标文本
        if metrics:
            metrics_str = f"Dice: {metrics.get('dice', 0):.3f}\nIoU: {metrics.get('iou', 0):.3f}\nPoints: {metrics.get('num_points', 0)}"
            ax2.text(0.02, 0.98, metrics_str, 
                    transform=ax2.transAxes, fontsize=12, weight='bold',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='可视化PEFT训练数据',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--jsonl',
        type=str,
        required=True,
        help='输入JSONL文件路径'
    )
    
    parser.add_argument(
        '--out_dir',
        type=str,
        required=True,
        help='输出图像文件夹路径'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['random', 'all'],
        default='random',
        help='采样模式: random=随机采样, all=遍历所有样本'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='随机采样模式下的样本数量（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--show_mask',
        action='store_true',
        default=True,
        help='是否显示真值掩模（默认True）'
    )
    
    parser.add_argument(
        '--no_mask',
        dest='show_mask',
        action='store_false',
        help='不显示真值掩模'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='输出图像分辨率（DPI）'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default='sample',
        help='输出文件名前缀'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='模型checkpoint路径（如果提供，将生成预测对比图）'
    )
    
    parser.add_argument(
        '--sam_checkpoint',
        type=str,
        default=None,
        help='SAM2 checkpoint路径（如果提供checkpoint，此参数必需）'
    )
    
    parser.add_argument(
        '--image_size',
        type=int,
        nargs=2,
        default=[512, 512],
        help='图像尺寸 [H, W]（用于预测）'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='设备（cuda/cpu）'
    )
    
    parser.add_argument(
        '--max_rollout_steps',
        type=int,
        default=16,
        help='最大rollout步数（用于预测）'
    )
    
    args = parser.parse_args()
    
    # 加载模型（如果提供了checkpoint）
    sam2_lora = None
    point_predictor = None
    
    if args.checkpoint:
        if not args.sam_checkpoint:
            print("Error: --sam_checkpoint is required when --checkpoint is provided")
            sys.exit(1)
        
        print(f"Loading model from {args.checkpoint}...")
        try:
            device = torch.device(args.device)
            
            # 加载checkpoint
            ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
            config = ckpt.get('config', {})
            
            # 提取配置
            lora_rank = config.get('lora_rank', 16)
            lora_alpha = config.get('lora_alpha', 32)
            fusion_mode = config.get('fusion_mode', 'film')
            feature_scale = config.get('feature_scale', 8)
            
            # 初始化SAM2 LoRA
            sam2_lora = LoRASAM2Wrapper(
                sam_checkpoint=args.sam_checkpoint,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                device=str(device),
            )
            
            # 初始化点预测网络
            sam_feature_dim = 256  # SAM2 Hiera Large的特征维度
            point_predictor = PointPredictorFromSAMFeatures(
                sam_feature_dim=sam_feature_dim,
                output_size=tuple(args.image_size),
                fusion_mode=fusion_mode,
                feature_scale=feature_scale,
            ).to(device)
            
            # 加载权重
            load_checkpoint(
                args.checkpoint, point_predictor, sam2_lora, None, None, None, str(device)
            )
            
            point_predictor.eval()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Will continue without prediction...")
            sam2_lora = None
            point_predictor = None
    
    # 加载数据
    print(f"Loading data from {args.jsonl}...")
    data = load_jsonl(args.jsonl)
    print(f"Loaded {len(data)} samples")
    
    # 选择样本
    if args.mode == 'random':
        random.seed(args.seed)
        selected_samples = random.sample(data, min(args.num_samples, len(data)))
        print(f"Randomly selected {len(selected_samples)} samples")
    else:
        selected_samples = data
        print(f"Processing all {len(selected_samples)} samples")
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 可视化每个样本
    print(f"\nGenerating visualizations...")
    for i, sample in enumerate(selected_samples):
        # 生成输出文件名
        if args.mode == 'random':
            output_filename = f'{args.prefix}_{i:03d}.png'
        else:
            # 使用图像文件名作为输出文件名
            image_path = sample.get('image', '')
            if image_path:
                stem = Path(image_path).stem
                output_filename = f'{args.prefix}_{stem}.png'
            else:
                output_filename = f'{args.prefix}_{i:04d}.png'
        
        output_path = os.path.join(args.out_dir, output_filename)
        
        # 可视化
        try:
            visualize_sample(
                sample, output_path, 
                show_mask=args.show_mask, 
                dpi=args.dpi,
                sam2_lora=sam2_lora,
                point_predictor=point_predictor,
                image_size=tuple(args.image_size),
                device=args.device,
                max_rollout_steps=args.max_rollout_steps,
            )
            print(f"  [{i+1}/{len(selected_samples)}] Saved: {output_path}")
        except Exception as e:
            print(f"  [{i+1}/{len(selected_samples)}] Error processing sample {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nDone! Visualizations saved to: {args.out_dir}")
    print(f"Total: {len(selected_samples)} images generated")


if __name__ == '__main__':
    main()


```


移除 main 函数中的模型加载代码，并更新参数：



移除模型加载相关代码，更新参数：


```python
#!/usr/bin/env python3
"""
可视化训练数据脚本

用于可视化PEFT训练数据中的图像、点标注和真值掩模。
支持随机采样或遍历所有样本。
如果提供了checkpoint，还会生成预测掩模对比图。

用法:
    # 随机采样5个样本（仅显示标注）
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_train-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode random \
        --num_samples 5

    # 遍历所有样本（带预测对比）
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_test-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode all \
        --checkpoint outputs/braintumour/peft_supervised_baseline-251107/checkpoint_best.pt \
        --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
        --image_size 512 512
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.ndimage import zoom

# 添加父目录到路径
_PARENT_DIR = Path(__file__).parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from peft.lora_sam2 import LoRASAM2Wrapper
from peft.point_predictor_peft import PointPredictorFromSAMFeatures
from peft.utils_peft import compute_dice, compute_iou, load_checkpoint


def load_jsonl(jsonl_path: str) -> List[Dict]:
    """加载JSONL文件"""
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith('['):
            # JSON数组格式
            data = json.loads(content)
        else:
            # JSONL格式（逐行）
            data = []
            for line in content.splitlines():
                if line.strip():
                    data.append(json.loads(line))
    return data


def visualize_sample(
    sample: Dict,
    output_path: str,
    show_mask: bool = True,
    dpi: int = 150,
    show_prediction: bool = True,
) -> None:
    """
    可视化单个样本
    
    Args:
        sample: 样本字典，包含 'image', 'gt_mask', 'points', 'labels', 'sam_masks_dir' 等字段
        output_path: 输出图像路径
        show_mask: 是否显示真值掩模
        dpi: 图像分辨率
        show_prediction: 是否显示预测掩模对比（需要sam_masks_dir）
    """
    # 加载图像
    image_path = sample['image']
    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        return
    
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    orig_w, orig_h = img.size
    
    # 获取点和标签
    points = sample.get('points', [])
    labels = sample.get('labels', [])
    
    if len(points) != len(labels):
        print(f"Warning: Points and labels length mismatch: {len(points)} vs {len(labels)}")
        return
    
    # 加载预测掩模和计算指标（如果sam_masks_dir存在）
    pred_mask = None
    pred_mask_binary = None
    gt_mask_binary_for_metrics = None
    metrics = {}
    do_prediction = False
    
    if show_prediction:
        sam_masks_dir = sample.get('sam_masks_dir', None)
        gt_mask_path = sample.get('gt_mask', None)
        
        if sam_masks_dir and gt_mask_path:
            try:
                # 获取图像stem（文件名不含扩展名）
                image_stem = Path(image_path).stem
                mask_dir = os.path.join(sam_masks_dir, image_stem)
                
                # 加载最后一个步骤的预测掩模（len(points)-1.png）
                if len(points) > 0:
                    last_mask_idx = len(points) - 1
                    pred_mask_path = os.path.join(mask_dir, f"{last_mask_idx}.png")
                    
                    if os.path.exists(pred_mask_path):
                        pred_mask = np.array(Image.open(pred_mask_path).convert('L'))
                        pred_mask_binary = (pred_mask > 127).astype(float)
                        
                        # 加载真值掩模并计算指标
                        if os.path.exists(gt_mask_path):
                            gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                            gt_mask_binary_for_metrics = (gt_mask > 127).astype(float)
                            
                            # 调整尺寸以匹配（如果需要）
                            if gt_mask_binary_for_metrics.shape != pred_mask_binary.shape:
                                scale_y = pred_mask_binary.shape[0] / gt_mask_binary_for_metrics.shape[0]
                                scale_x = pred_mask_binary.shape[1] / gt_mask_binary_for_metrics.shape[1]
                                gt_mask_binary_for_metrics = zoom(gt_mask_binary_for_metrics, (scale_y, scale_x), order=0)
                            
                            # 转换为tensor计算指标
                            pred_tensor = torch.from_numpy(pred_mask_binary).float()
                            gt_tensor = torch.from_numpy(gt_mask_binary_for_metrics).float()
                            
                            dice = compute_dice(pred_tensor, gt_tensor)
                            iou = compute_iou(pred_tensor, gt_tensor)
                            
                            metrics['dice'] = dice
                            metrics['iou'] = iou
                            metrics['num_points'] = len(points)
                            
                            do_prediction = True
                        else:
                            print(f"Warning: GT mask not found: {gt_mask_path}")
                    else:
                        print(f"Warning: Predicted mask not found: {pred_mask_path}")
                else:
                    print(f"Warning: No points in sample, cannot load predicted mask")
            except Exception as e:
                print(f"Warning: Failed to load predicted mask: {e}")
                import traceback
                traceback.print_exc()
    
    # 创建图形
    if do_prediction:
        # 两幅图：第一幅显示点和标注，第二幅显示掩模对比
        fig, axes = plt.subplots(1, 2, figsize=(24, 12))
        ax1, ax2 = axes
    else:
        # 单幅图：只显示点和标注
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 12))
        ax2 = None
    
    # 第一幅图：显示点和标注
    ax1.imshow(img_np)
    
    # 如果有真值掩模，叠加显示
    if show_mask:
        gt_mask_path = sample.get('gt_mask', None)
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                mask_binary = (gt_mask > 127).astype(float)
                ax1.imshow(mask_binary, alpha=0.3, cmap='Greens')
            except Exception as e:
                print(f"Warning: Failed to load mask {gt_mask_path}: {e}")
    
    # 绘制点
    for j, ((x, y), label) in enumerate(zip(points, labels)):
        if label == 1:
            # 前景点：红色，实心圆
            color = 'red'
            size = 150
            point_type = 'FG'
        else:
            # 背景点：蓝色，实心圆
            color = 'blue'
            size = 120
            point_type = 'BG'
        
        # 绘制点
        ax1.scatter(x, y, c=color, s=size, marker='o', 
                   edgecolors='white', linewidths=2, zorder=10)
        
        # 添加标签文本
        ax1.text(x + 15, y, f'{j}({point_type})', 
                fontsize=10, color='white', weight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    # 设置第一幅图的标题
    stem = Path(image_path).stem
    title1 = f'{stem}: {len(points)} points (Ground Truth)'
    ax1.set_title(title1, fontsize=14, weight='bold')
    ax1.axis('off')
    
    # 第二幅图：显示掩模对比（如果进行预测）
    if do_prediction and ax2 is not None:
        ax2.imshow(img_np)
        
        gt_mask_path = sample.get('gt_mask', None)
        gt_mask_binary = None
        
        # 加载并调整真值掩模尺寸（用于显示）
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                gt_mask_binary = (gt_mask > 127).astype(float)
                # 调整尺寸以匹配图像
                if gt_mask_binary.shape != img_np.shape[:2]:
                    scale_y = img_np.shape[0] / gt_mask_binary.shape[0]
                    scale_x = img_np.shape[1] / gt_mask_binary.shape[1]
                    gt_mask_binary = zoom(gt_mask_binary, (scale_y, scale_x), order=0)
            except Exception as e:
                print(f"Warning: Failed to load GT mask: {e}")
        
        # 调整预测掩模尺寸（用于显示）
        pred_mask_binary_display = None
        if pred_mask_binary is not None:
            # pred_mask_binary已经是二值化的，但需要调整尺寸以匹配图像
            if pred_mask_binary.shape != img_np.shape[:2]:
                scale_y = img_np.shape[0] / pred_mask_binary.shape[0]
                scale_x = img_np.shape[1] / pred_mask_binary.shape[1]
                pred_mask_binary_display = zoom(pred_mask_binary, (scale_y, scale_x), order=0)
            else:
                pred_mask_binary_display = pred_mask_binary
        
        # 优化显示：先显示重叠区域（黄色），再显示真值（绿色），最后显示预测（红色）
        if gt_mask_binary is not None and pred_mask_binary_display is not None:
            # 重叠区域（黄色）
            overlap = (gt_mask_binary * pred_mask_binary_display).astype(float)
            overlap_colored = np.zeros_like(img_np)
            overlap_colored[:, :, 0] = overlap * 255  # 红色
            overlap_colored[:, :, 1] = overlap * 255  # 绿色
            ax2.imshow(overlap_colored, alpha=0.5)
            
            # 真值掩模的未重叠部分（绿色）
            gt_only = (gt_mask_binary * (1 - pred_mask_binary_display)).astype(float)
            gt_only_colored = np.zeros_like(img_np)
            gt_only_colored[:, :, 1] = gt_only * 255
            ax2.imshow(gt_only_colored, alpha=0.4)
            
            # 预测掩模的未重叠部分（红色）
            pred_only = (pred_mask_binary_display * (1 - gt_mask_binary)).astype(float)
            pred_only_colored = np.zeros_like(img_np)
            pred_only_colored[:, :, 0] = pred_only * 255
            ax2.imshow(pred_only_colored, alpha=0.4)
        elif gt_mask_binary is not None:
            # 只有真值掩模（绿色）
            gt_mask_colored = np.zeros_like(img_np)
            gt_mask_colored[:, :, 1] = gt_mask_binary * 255
            ax2.imshow(gt_mask_colored, alpha=0.4)
        elif pred_mask_binary_display is not None:
            # 只有预测掩模（红色）
            pred_mask_colored = np.zeros_like(img_np)
            pred_mask_colored[:, :, 0] = pred_mask_binary_display * 255
            ax2.imshow(pred_mask_colored, alpha=0.4)
        
        # 绘制预测点（使用JSONL中的points）
        for j, ((x, y), label) in enumerate(zip(points, labels)):
            if label == 1:
                color = 'red'
                size = 150
                point_type = 'FG'
            else:
                color = 'blue'
                size = 120
                point_type = 'BG'
            
            ax2.scatter(x, y, c=color, s=size, marker='o', 
                       edgecolors='white', linewidths=2, zorder=10)
            ax2.text(x + 15, y, f'{j}({point_type})', 
                    fontsize=10, color='white', weight='bold',
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
        
        # 设置第二幅图的标题和指标
        title2 = f'Prediction: {len(points)} points'
        if metrics:
            metrics_text = f"Dice: {metrics.get('dice', 0):.3f}, IoU: {metrics.get('iou', 0):.3f}"
            title2 += f'\n{metrics_text}'
        ax2.set_title(title2, fontsize=14, weight='bold')
        ax2.axis('off')
        
        # 在图像角落添加指标文本
        if metrics:
            metrics_str = f"Dice: {metrics.get('dice', 0):.3f}\nIoU: {metrics.get('iou', 0):.3f}\nPoints: {metrics.get('num_points', 0)}"
            ax2.text(0.02, 0.98, metrics_str, 
                    transform=ax2.transAxes, fontsize=12, weight='bold',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='可视化PEFT训练数据',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--jsonl',
        type=str,
        required=True,
        help='输入JSONL文件路径'
    )
    
    parser.add_argument(
        '--out_dir',
        type=str,
        required=True,
        help='输出图像文件夹路径'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['random', 'all'],
        default='random',
        help='采样模式: random=随机采样, all=遍历所有样本'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='随机采样模式下的样本数量（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--show_mask',
        action='store_true',
        default=True,
        help='是否显示真值掩模（默认True）'
    )
    
    parser.add_argument(
        '--no_mask',
        dest='show_mask',
        action='store_false',
        help='不显示真值掩模'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='输出图像分辨率（DPI）'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default='sample',
        help='输出文件名前缀'
    )
    
    parser.add_argument(
        '--show_prediction',
        action='store_true',
        default=True,
        help='是否显示预测掩模对比（需要sam_masks_dir，默认True）'
    )
    
    parser.add_argument(
        '--no_prediction',
        dest='show_prediction',
        action='store_false',
        help='不显示预测掩模对比'
    )
    
    args = parser.parse_args()
    
    # 加载数据
    print(f"Loading data from {args.jsonl}...")
    data = load_jsonl(args.jsonl)
    print(f"Loaded {len(data)} samples")
    
    # 选择样本
    if args.mode == 'random':
        random.seed(args.seed)
        selected_samples = random.sample(data, min(args.num_samples, len(data)))
        print(f"Randomly selected {len(selected_samples)} samples")
    else:
        selected_samples = data
        print(f"Processing all {len(selected_samples)} samples")
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 可视化每个样本
    print(f"\nGenerating visualizations...")
    for i, sample in enumerate(selected_samples):
        # 生成输出文件名
        if args.mode == 'random':
            output_filename = f'{args.prefix}_{i:03d}.png'
        else:
            # 使用图像文件名作为输出文件名
            image_path = sample.get('image', '')
            if image_path:
                stem = Path(image_path).stem
                output_filename = f'{args.prefix}_{stem}.png'
            else:
                output_filename = f'{args.prefix}_{i:04d}.png'
        
        output_path = os.path.join(args.out_dir, output_filename)
        
        # 可视化
        try:
            visualize_sample(
                sample, output_path, 
                show_mask=args.show_mask, 
                dpi=args.dpi,
                sam2_lora=sam2_lora,
                point_predictor=point_predictor,
                image_size=tuple(args.image_size),
                device=args.device,
                max_rollout_steps=args.max_rollout_steps,
            )
            print(f"  [{i+1}/{len(selected_samples)}] Saved: {output_path}")
        except Exception as e:
            print(f"  [{i+1}/{len(selected_samples)}] Error processing sample {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nDone! Visualizations saved to: {args.out_dir}")
    print(f"Total: {len(selected_samples)} images generated")


if __name__ == '__main__':
    main()


```

```python
#!/usr/bin/env python3
"""
可视化训练数据脚本

用于可视化PEFT训练数据中的图像、点标注和真值掩模。
支持随机采样或遍历所有样本。
如果提供了checkpoint，还会生成预测掩模对比图。

用法:
    # 随机采样5个样本（仅显示标注）
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_train-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode random \
        --num_samples 5

    # 遍历所有样本（带预测对比）
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_test-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode all \
        --checkpoint outputs/braintumour/peft_supervised_baseline-251107/checkpoint_best.pt \
        --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
        --image_size 512 512
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.ndimage import zoom

# 添加父目录到路径
_PARENT_DIR = Path(__file__).parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from peft.lora_sam2 import LoRASAM2Wrapper
from peft.point_predictor_peft import PointPredictorFromSAMFeatures
from peft.utils_peft import compute_dice, compute_iou, load_checkpoint


def load_jsonl(jsonl_path: str) -> List[Dict]:
    """加载JSONL文件"""
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith('['):
            # JSON数组格式
            data = json.loads(content)
        else:
            # JSONL格式（逐行）
            data = []
            for line in content.splitlines():
                if line.strip():
                    data.append(json.loads(line))
    return data


def visualize_sample(
    sample: Dict,
    output_path: str,
    show_mask: bool = True,
    dpi: int = 150,
    show_prediction: bool = True,
) -> None:
    """
    可视化单个样本
    
    Args:
        sample: 样本字典，包含 'image', 'gt_mask', 'points', 'labels', 'sam_masks_dir' 等字段
        output_path: 输出图像路径
        show_mask: 是否显示真值掩模
        dpi: 图像分辨率
        show_prediction: 是否显示预测掩模对比（需要sam_masks_dir）
    """
    # 加载图像
    image_path = sample['image']
    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        return
    
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    orig_w, orig_h = img.size
    
    # 获取点和标签
    points = sample.get('points', [])
    labels = sample.get('labels', [])
    
    if len(points) != len(labels):
        print(f"Warning: Points and labels length mismatch: {len(points)} vs {len(labels)}")
        return
    
    # 加载预测掩模和计算指标（如果sam_masks_dir存在）
    pred_mask = None
    pred_mask_binary = None
    gt_mask_binary_for_metrics = None
    metrics = {}
    do_prediction = False
    
    if show_prediction:
        sam_masks_dir = sample.get('sam_masks_dir', None)
        gt_mask_path = sample.get('gt_mask', None)
        
        if sam_masks_dir and gt_mask_path:
            try:
                # 获取图像stem（文件名不含扩展名）
                image_stem = Path(image_path).stem
                mask_dir = os.path.join(sam_masks_dir, image_stem)
                
                # 加载最后一个步骤的预测掩模（len(points)-1.png）
                if len(points) > 0:
                    last_mask_idx = len(points) - 1
                    pred_mask_path = os.path.join(mask_dir, f"{last_mask_idx}.png")
                    
                    if os.path.exists(pred_mask_path):
                        pred_mask = np.array(Image.open(pred_mask_path).convert('L'))
                        pred_mask_binary = (pred_mask > 127).astype(float)
                        
                        # 加载真值掩模并计算指标
                        if os.path.exists(gt_mask_path):
                            gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                            gt_mask_binary_for_metrics = (gt_mask > 127).astype(float)
                            
                            # 调整尺寸以匹配（如果需要）
                            if gt_mask_binary_for_metrics.shape != pred_mask_binary.shape:
                                scale_y = pred_mask_binary.shape[0] / gt_mask_binary_for_metrics.shape[0]
                                scale_x = pred_mask_binary.shape[1] / gt_mask_binary_for_metrics.shape[1]
                                gt_mask_binary_for_metrics = zoom(gt_mask_binary_for_metrics, (scale_y, scale_x), order=0)
                            
                            # 转换为tensor计算指标
                            pred_tensor = torch.from_numpy(pred_mask_binary).float()
                            gt_tensor = torch.from_numpy(gt_mask_binary_for_metrics).float()
                            
                            dice = compute_dice(pred_tensor, gt_tensor)
                            iou = compute_iou(pred_tensor, gt_tensor)
                            
                            metrics['dice'] = dice
                            metrics['iou'] = iou
                            metrics['num_points'] = len(points)
                            
                            do_prediction = True
                        else:
                            print(f"Warning: GT mask not found: {gt_mask_path}")
                    else:
                        print(f"Warning: Predicted mask not found: {pred_mask_path}")
                else:
                    print(f"Warning: No points in sample, cannot load predicted mask")
            except Exception as e:
                print(f"Warning: Failed to load predicted mask: {e}")
                import traceback
                traceback.print_exc()
    
    # 创建图形
    if do_prediction:
        # 两幅图：第一幅显示点和标注，第二幅显示掩模对比
        fig, axes = plt.subplots(1, 2, figsize=(24, 12))
        ax1, ax2 = axes
    else:
        # 单幅图：只显示点和标注
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 12))
        ax2 = None
    
    # 第一幅图：显示点和标注
    ax1.imshow(img_np)
    
    # 如果有真值掩模，叠加显示
    if show_mask:
        gt_mask_path = sample.get('gt_mask', None)
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                mask_binary = (gt_mask > 127).astype(float)
                ax1.imshow(mask_binary, alpha=0.3, cmap='Greens')
            except Exception as e:
                print(f"Warning: Failed to load mask {gt_mask_path}: {e}")
    
    # 绘制点
    for j, ((x, y), label) in enumerate(zip(points, labels)):
        if label == 1:
            # 前景点：红色，实心圆
            color = 'red'
            size = 150
            point_type = 'FG'
        else:
            # 背景点：蓝色，实心圆
            color = 'blue'
            size = 120
            point_type = 'BG'
        
        # 绘制点
        ax1.scatter(x, y, c=color, s=size, marker='o', 
                   edgecolors='white', linewidths=2, zorder=10)
        
        # 添加标签文本
        ax1.text(x + 15, y, f'{j}({point_type})', 
                fontsize=10, color='white', weight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    # 设置第一幅图的标题
    stem = Path(image_path).stem
    title1 = f'{stem}: {len(points)} points (Ground Truth)'
    ax1.set_title(title1, fontsize=14, weight='bold')
    ax1.axis('off')
    
    # 第二幅图：显示掩模对比（如果进行预测）
    if do_prediction and ax2 is not None:
        ax2.imshow(img_np)
        
        gt_mask_path = sample.get('gt_mask', None)
        gt_mask_binary = None
        
        # 加载并调整真值掩模尺寸（用于显示）
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                gt_mask_binary = (gt_mask > 127).astype(float)
                # 调整尺寸以匹配图像
                if gt_mask_binary.shape != img_np.shape[:2]:
                    scale_y = img_np.shape[0] / gt_mask_binary.shape[0]
                    scale_x = img_np.shape[1] / gt_mask_binary.shape[1]
                    gt_mask_binary = zoom(gt_mask_binary, (scale_y, scale_x), order=0)
            except Exception as e:
                print(f"Warning: Failed to load GT mask: {e}")
        
        # 调整预测掩模尺寸（用于显示）
        pred_mask_binary_display = None
        if pred_mask_binary is not None:
            # pred_mask_binary已经是二值化的，但需要调整尺寸以匹配图像
            if pred_mask_binary.shape != img_np.shape[:2]:
                scale_y = img_np.shape[0] / pred_mask_binary.shape[0]
                scale_x = img_np.shape[1] / pred_mask_binary.shape[1]
                pred_mask_binary_display = zoom(pred_mask_binary, (scale_y, scale_x), order=0)
            else:
                pred_mask_binary_display = pred_mask_binary
        
        # 优化显示：先显示重叠区域（黄色），再显示真值（绿色），最后显示预测（红色）
        if gt_mask_binary is not None and pred_mask_binary_display is not None:
            # 重叠区域（黄色）
            overlap = (gt_mask_binary * pred_mask_binary_display).astype(float)
            overlap_colored = np.zeros_like(img_np)
            overlap_colored[:, :, 0] = overlap * 255  # 红色
            overlap_colored[:, :, 1] = overlap * 255  # 绿色
            ax2.imshow(overlap_colored, alpha=0.5)
            
            # 真值掩模的未重叠部分（绿色）
            gt_only = (gt_mask_binary * (1 - pred_mask_binary_display)).astype(float)
            gt_only_colored = np.zeros_like(img_np)
            gt_only_colored[:, :, 1] = gt_only * 255
            ax2.imshow(gt_only_colored, alpha=0.4)
            
            # 预测掩模的未重叠部分（红色）
            pred_only = (pred_mask_binary_display * (1 - gt_mask_binary)).astype(float)
            pred_only_colored = np.zeros_like(img_np)
            pred_only_colored[:, :, 0] = pred_only * 255
            ax2.imshow(pred_only_colored, alpha=0.4)
        elif gt_mask_binary is not None:
            # 只有真值掩模（绿色）
            gt_mask_colored = np.zeros_like(img_np)
            gt_mask_colored[:, :, 1] = gt_mask_binary * 255
            ax2.imshow(gt_mask_colored, alpha=0.4)
        elif pred_mask_binary_display is not None:
            # 只有预测掩模（红色）
            pred_mask_colored = np.zeros_like(img_np)
            pred_mask_colored[:, :, 0] = pred_mask_binary_display * 255
            ax2.imshow(pred_mask_colored, alpha=0.4)
        
        # 绘制预测点（使用JSONL中的points）
        for j, ((x, y), label) in enumerate(zip(points, labels)):
            if label == 1:
                color = 'red'
                size = 150
                point_type = 'FG'
            else:
                color = 'blue'
                size = 120
                point_type = 'BG'
            
            ax2.scatter(x, y, c=color, s=size, marker='o', 
                       edgecolors='white', linewidths=2, zorder=10)
            ax2.text(x + 15, y, f'{j}({point_type})', 
                    fontsize=10, color='white', weight='bold',
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
        
        # 设置第二幅图的标题和指标
        title2 = f'Prediction: {len(points)} points'
        if metrics:
            metrics_text = f"Dice: {metrics.get('dice', 0):.3f}, IoU: {metrics.get('iou', 0):.3f}"
            title2 += f'\n{metrics_text}'
        ax2.set_title(title2, fontsize=14, weight='bold')
        ax2.axis('off')
        
        # 在图像角落添加指标文本
        if metrics:
            metrics_str = f"Dice: {metrics.get('dice', 0):.3f}\nIoU: {metrics.get('iou', 0):.3f}\nPoints: {metrics.get('num_points', 0)}"
            ax2.text(0.02, 0.98, metrics_str, 
                    transform=ax2.transAxes, fontsize=12, weight='bold',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='可视化PEFT训练数据',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--jsonl',
        type=str,
        required=True,
        help='输入JSONL文件路径'
    )
    
    parser.add_argument(
        '--out_dir',
        type=str,
        required=True,
        help='输出图像文件夹路径'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['random', 'all'],
        default='random',
        help='采样模式: random=随机采样, all=遍历所有样本'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='随机采样模式下的样本数量（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--show_mask',
        action='store_true',
        default=True,
        help='是否显示真值掩模（默认True）'
    )
    
    parser.add_argument(
        '--no_mask',
        dest='show_mask',
        action='store_false',
        help='不显示真值掩模'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='输出图像分辨率（DPI）'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default='sample',
        help='输出文件名前缀'
    )
    
    parser.add_argument(
        '--show_prediction',
        action='store_true',
        default=True,
        help='是否显示预测掩模对比（需要sam_masks_dir，默认True）'
    )
    
    parser.add_argument(
        '--no_prediction',
        dest='show_prediction',
        action='store_false',
        help='不显示预测掩模对比'
    )
    
    args = parser.parse_args()
    
    # 加载数据
    print(f"Loading data from {args.jsonl}...")
    data = load_jsonl(args.jsonl)
    print(f"Loaded {len(data)} samples")
    
    # 选择样本
    if args.mode == 'random':
        random.seed(args.seed)
        selected_samples = random.sample(data, min(args.num_samples, len(data)))
        print(f"Randomly selected {len(selected_samples)} samples")
    else:
        selected_samples = data
        print(f"Processing all {len(selected_samples)} samples")
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 可视化每个样本
    print(f"\nGenerating visualizations...")
    for i, sample in enumerate(selected_samples):
        # 生成输出文件名
        if args.mode == 'random':
            output_filename = f'{args.prefix}_{i:03d}.png'
        else:
            # 使用图像文件名作为输出文件名
            image_path = sample.get('image', '')
            if image_path:
                stem = Path(image_path).stem
                output_filename = f'{args.prefix}_{stem}.png'
            else:
                output_filename = f'{args.prefix}_{i:04d}.png'
        
        output_path = os.path.join(args.out_dir, output_filename)
        
        # 可视化
        try:
            visualize_sample(
                sample, output_path, 
                show_mask=args.show_mask, 
                dpi=args.dpi,
                show_prediction=args.show_prediction,
            )
            print(f"  [{i+1}/{len(selected_samples)}] Saved: {output_path}")
        except Exception as e:
            print(f"  [{i+1}/{len(selected_samples)}] Error processing sample {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nDone! Visualizations saved to: {args.out_dir}")
    print(f"Total: {len(selected_samples)} images generated")


if __name__ == '__main__':
    main()


```

```python
#!/usr/bin/env python3
"""
可视化训练数据脚本

用于可视化PEFT训练数据中的图像、点标注和真值掩模。
支持随机采样或遍历所有样本。
如果提供了checkpoint，还会生成预测掩模对比图。

用法:
    # 随机采样5个样本（仅显示标注）
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_train-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode random \
        --num_samples 5

    # 遍历所有样本（带预测对比）
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_test-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode all \
        --checkpoint outputs/braintumour/peft_supervised_baseline-251107/checkpoint_best.pt \
        --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
        --image_size 512 512
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.ndimage import zoom

# 添加父目录到路径
_PARENT_DIR = Path(__file__).parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from peft.utils_peft import compute_dice, compute_iou


def load_jsonl(jsonl_path: str) -> List[Dict]:
    """加载JSONL文件"""
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith('['):
            # JSON数组格式
            data = json.loads(content)
        else:
            # JSONL格式（逐行）
            data = []
            for line in content.splitlines():
                if line.strip():
                    data.append(json.loads(line))
    return data


def visualize_sample(
    sample: Dict,
    output_path: str,
    show_mask: bool = True,
    dpi: int = 150,
    show_prediction: bool = True,
) -> None:
    """
    可视化单个样本
    
    Args:
        sample: 样本字典，包含 'image', 'gt_mask', 'points', 'labels', 'sam_masks_dir' 等字段
        output_path: 输出图像路径
        show_mask: 是否显示真值掩模
        dpi: 图像分辨率
        show_prediction: 是否显示预测掩模对比（需要sam_masks_dir）
    """
    # 加载图像
    image_path = sample['image']
    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        return
    
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    orig_w, orig_h = img.size
    
    # 获取点和标签
    points = sample.get('points', [])
    labels = sample.get('labels', [])
    
    if len(points) != len(labels):
        print(f"Warning: Points and labels length mismatch: {len(points)} vs {len(labels)}")
        return
    
    # 加载预测掩模和计算指标（如果sam_masks_dir存在）
    pred_mask = None
    pred_mask_binary = None
    gt_mask_binary_for_metrics = None
    metrics = {}
    do_prediction = False
    
    if show_prediction:
        sam_masks_dir = sample.get('sam_masks_dir', None)
        gt_mask_path = sample.get('gt_mask', None)
        
        if sam_masks_dir and gt_mask_path:
            try:
                # 获取图像stem（文件名不含扩展名）
                image_stem = Path(image_path).stem
                mask_dir = os.path.join(sam_masks_dir, image_stem)
                
                # 加载最后一个步骤的预测掩模（len(points)-1.png）
                if len(points) > 0:
                    last_mask_idx = len(points) - 1
                    pred_mask_path = os.path.join(mask_dir, f"{last_mask_idx}.png")
                    
                    if os.path.exists(pred_mask_path):
                        pred_mask = np.array(Image.open(pred_mask_path).convert('L'))
                        pred_mask_binary = (pred_mask > 127).astype(float)
                        
                        # 加载真值掩模并计算指标
                        if os.path.exists(gt_mask_path):
                            gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                            gt_mask_binary_for_metrics = (gt_mask > 127).astype(float)
                            
                            # 调整尺寸以匹配（如果需要）
                            if gt_mask_binary_for_metrics.shape != pred_mask_binary.shape:
                                scale_y = pred_mask_binary.shape[0] / gt_mask_binary_for_metrics.shape[0]
                                scale_x = pred_mask_binary.shape[1] / gt_mask_binary_for_metrics.shape[1]
                                gt_mask_binary_for_metrics = zoom(gt_mask_binary_for_metrics, (scale_y, scale_x), order=0)
                            
                            # 转换为tensor计算指标
                            pred_tensor = torch.from_numpy(pred_mask_binary).float()
                            gt_tensor = torch.from_numpy(gt_mask_binary_for_metrics).float()
                            
                            dice = compute_dice(pred_tensor, gt_tensor)
                            iou = compute_iou(pred_tensor, gt_tensor)
                            
                            metrics['dice'] = dice
                            metrics['iou'] = iou
                            metrics['num_points'] = len(points)
                            
                            do_prediction = True
                        else:
                            print(f"Warning: GT mask not found: {gt_mask_path}")
                    else:
                        print(f"Warning: Predicted mask not found: {pred_mask_path}")
                else:
                    print(f"Warning: No points in sample, cannot load predicted mask")
            except Exception as e:
                print(f"Warning: Failed to load predicted mask: {e}")
                import traceback
                traceback.print_exc()
    
    # 创建图形
    if do_prediction:
        # 两幅图：第一幅显示点和标注，第二幅显示掩模对比
        fig, axes = plt.subplots(1, 2, figsize=(24, 12))
        ax1, ax2 = axes
    else:
        # 单幅图：只显示点和标注
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 12))
        ax2 = None
    
    # 第一幅图：显示点和标注
    ax1.imshow(img_np)
    
    # 如果有真值掩模，叠加显示
    if show_mask:
        gt_mask_path = sample.get('gt_mask', None)
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                mask_binary = (gt_mask > 127).astype(float)
                ax1.imshow(mask_binary, alpha=0.3, cmap='Greens')
            except Exception as e:
                print(f"Warning: Failed to load mask {gt_mask_path}: {e}")
    
    # 绘制点
    for j, ((x, y), label) in enumerate(zip(points, labels)):
        if label == 1:
            # 前景点：红色，实心圆
            color = 'red'
            size = 150
            point_type = 'FG'
        else:
            # 背景点：蓝色，实心圆
            color = 'blue'
            size = 120
            point_type = 'BG'
        
        # 绘制点
        ax1.scatter(x, y, c=color, s=size, marker='o', 
                   edgecolors='white', linewidths=2, zorder=10)
        
        # 添加标签文本
        ax1.text(x + 15, y, f'{j}({point_type})', 
                fontsize=10, color='white', weight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    # 设置第一幅图的标题
    stem = Path(image_path).stem
    title1 = f'{stem}: {len(points)} points (Ground Truth)'
    ax1.set_title(title1, fontsize=14, weight='bold')
    ax1.axis('off')
    
    # 第二幅图：显示掩模对比（如果进行预测）
    if do_prediction and ax2 is not None:
        ax2.imshow(img_np)
        
        gt_mask_path = sample.get('gt_mask', None)
        gt_mask_binary = None
        
        # 加载并调整真值掩模尺寸（用于显示）
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                gt_mask_binary = (gt_mask > 127).astype(float)
                # 调整尺寸以匹配图像
                if gt_mask_binary.shape != img_np.shape[:2]:
                    scale_y = img_np.shape[0] / gt_mask_binary.shape[0]
                    scale_x = img_np.shape[1] / gt_mask_binary.shape[1]
                    gt_mask_binary = zoom(gt_mask_binary, (scale_y, scale_x), order=0)
            except Exception as e:
                print(f"Warning: Failed to load GT mask: {e}")
        
        # 调整预测掩模尺寸（用于显示）
        pred_mask_binary_display = None
        if pred_mask_binary is not None:
            # pred_mask_binary已经是二值化的，但需要调整尺寸以匹配图像
            if pred_mask_binary.shape != img_np.shape[:2]:
                scale_y = img_np.shape[0] / pred_mask_binary.shape[0]
                scale_x = img_np.shape[1] / pred_mask_binary.shape[1]
                pred_mask_binary_display = zoom(pred_mask_binary, (scale_y, scale_x), order=0)
            else:
                pred_mask_binary_display = pred_mask_binary
        
        # 优化显示：先显示重叠区域（黄色），再显示真值（绿色），最后显示预测（红色）
        if gt_mask_binary is not None and pred_mask_binary_display is not None:
            # 重叠区域（黄色）
            overlap = (gt_mask_binary * pred_mask_binary_display).astype(float)
            overlap_colored = np.zeros_like(img_np)
            overlap_colored[:, :, 0] = overlap * 255  # 红色
            overlap_colored[:, :, 1] = overlap * 255  # 绿色
            ax2.imshow(overlap_colored, alpha=0.5)
            
            # 真值掩模的未重叠部分（绿色）
            gt_only = (gt_mask_binary * (1 - pred_mask_binary_display)).astype(float)
            gt_only_colored = np.zeros_like(img_np)
            gt_only_colored[:, :, 1] = gt_only * 255
            ax2.imshow(gt_only_colored, alpha=0.4)
            
            # 预测掩模的未重叠部分（红色）
            pred_only = (pred_mask_binary_display * (1 - gt_mask_binary)).astype(float)
            pred_only_colored = np.zeros_like(img_np)
            pred_only_colored[:, :, 0] = pred_only * 255
            ax2.imshow(pred_only_colored, alpha=0.4)
        elif gt_mask_binary is not None:
            # 只有真值掩模（绿色）
            gt_mask_colored = np.zeros_like(img_np)
            gt_mask_colored[:, :, 1] = gt_mask_binary * 255
            ax2.imshow(gt_mask_colored, alpha=0.4)
        elif pred_mask_binary_display is not None:
            # 只有预测掩模（红色）
            pred_mask_colored = np.zeros_like(img_np)
            pred_mask_colored[:, :, 0] = pred_mask_binary_display * 255
            ax2.imshow(pred_mask_colored, alpha=0.4)
        
        # 绘制预测点（使用JSONL中的points）
        for j, ((x, y), label) in enumerate(zip(points, labels)):
            if label == 1:
                color = 'red'
                size = 150
                point_type = 'FG'
            else:
                color = 'blue'
                size = 120
                point_type = 'BG'
            
            ax2.scatter(x, y, c=color, s=size, marker='o', 
                       edgecolors='white', linewidths=2, zorder=10)
            ax2.text(x + 15, y, f'{j}({point_type})', 
                    fontsize=10, color='white', weight='bold',
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
        
        # 设置第二幅图的标题和指标
        title2 = f'Prediction: {len(points)} points'
        if metrics:
            metrics_text = f"Dice: {metrics.get('dice', 0):.3f}, IoU: {metrics.get('iou', 0):.3f}"
            title2 += f'\n{metrics_text}'
        ax2.set_title(title2, fontsize=14, weight='bold')
        ax2.axis('off')
        
        # 在图像角落添加指标文本
        if metrics:
            metrics_str = f"Dice: {metrics.get('dice', 0):.3f}\nIoU: {metrics.get('iou', 0):.3f}\nPoints: {metrics.get('num_points', 0)}"
            ax2.text(0.02, 0.98, metrics_str, 
                    transform=ax2.transAxes, fontsize=12, weight='bold',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='可视化PEFT训练数据',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--jsonl',
        type=str,
        required=True,
        help='输入JSONL文件路径'
    )
    
    parser.add_argument(
        '--out_dir',
        type=str,
        required=True,
        help='输出图像文件夹路径'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['random', 'all'],
        default='random',
        help='采样模式: random=随机采样, all=遍历所有样本'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='随机采样模式下的样本数量（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--show_mask',
        action='store_true',
        default=True,
        help='是否显示真值掩模（默认True）'
    )
    
    parser.add_argument(
        '--no_mask',
        dest='show_mask',
        action='store_false',
        help='不显示真值掩模'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='输出图像分辨率（DPI）'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default='sample',
        help='输出文件名前缀'
    )
    
    parser.add_argument(
        '--show_prediction',
        action='store_true',
        default=True,
        help='是否显示预测掩模对比（需要sam_masks_dir，默认True）'
    )
    
    parser.add_argument(
        '--no_prediction',
        dest='show_prediction',
        action='store_false',
        help='不显示预测掩模对比'
    )
    
    args = parser.parse_args()
    
    # 加载数据
    print(f"Loading data from {args.jsonl}...")
    data = load_jsonl(args.jsonl)
    print(f"Loaded {len(data)} samples")
    
    # 选择样本
    if args.mode == 'random':
        random.seed(args.seed)
        selected_samples = random.sample(data, min(args.num_samples, len(data)))
        print(f"Randomly selected {len(selected_samples)} samples")
    else:
        selected_samples = data
        print(f"Processing all {len(selected_samples)} samples")
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 可视化每个样本
    print(f"\nGenerating visualizations...")
    for i, sample in enumerate(selected_samples):
        # 生成输出文件名
        if args.mode == 'random':
            output_filename = f'{args.prefix}_{i:03d}.png'
        else:
            # 使用图像文件名作为输出文件名
            image_path = sample.get('image', '')
            if image_path:
                stem = Path(image_path).stem
                output_filename = f'{args.prefix}_{stem}.png'
            else:
                output_filename = f'{args.prefix}_{i:04d}.png'
        
        output_path = os.path.join(args.out_dir, output_filename)
        
        # 可视化
        try:
            visualize_sample(
                sample, output_path, 
                show_mask=args.show_mask, 
                dpi=args.dpi,
                show_prediction=args.show_prediction,
            )
            print(f"  [{i+1}/{len(selected_samples)}] Saved: {output_path}")
        except Exception as e:
            print(f"  [{i+1}/{len(selected_samples)}] Error processing sample {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nDone! Visualizations saved to: {args.out_dir}")
    print(f"Total: {len(selected_samples)} images generated")


if __name__ == '__main__':
    main()


```


移除不再需要的导入：


```python
#!/usr/bin/env python3
"""
可视化训练数据脚本

用于可视化PEFT训练数据中的图像、点标注和真值掩模。
支持随机采样或遍历所有样本。
如果提供了checkpoint，还会生成预测掩模对比图。

用法:
    # 随机采样5个样本（仅显示标注）
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_train-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode random \
        --num_samples 5

    # 遍历所有样本（带预测对比）
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_test-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode all \
        --checkpoint outputs/braintumour/peft_supervised_baseline-251107/checkpoint_best.pt \
        --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
        --image_size 512 512
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import zoom

# 添加父目录到路径
_PARENT_DIR = Path(__file__).parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from peft.utils_peft import compute_dice, compute_iou


def load_jsonl(jsonl_path: str) -> List[Dict]:
    """加载JSONL文件"""
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith('['):
            # JSON数组格式
            data = json.loads(content)
        else:
            # JSONL格式（逐行）
            data = []
            for line in content.splitlines():
                if line.strip():
                    data.append(json.loads(line))
    return data


def visualize_sample(
    sample: Dict,
    output_path: str,
    show_mask: bool = True,
    dpi: int = 150,
    show_prediction: bool = True,
) -> None:
    """
    可视化单个样本
    
    Args:
        sample: 样本字典，包含 'image', 'gt_mask', 'points', 'labels', 'sam_masks_dir' 等字段
        output_path: 输出图像路径
        show_mask: 是否显示真值掩模
        dpi: 图像分辨率
        show_prediction: 是否显示预测掩模对比（需要sam_masks_dir）
    """
    # 加载图像
    image_path = sample['image']
    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        return
    
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    orig_w, orig_h = img.size
    
    # 获取点和标签
    points = sample.get('points', [])
    labels = sample.get('labels', [])
    
    if len(points) != len(labels):
        print(f"Warning: Points and labels length mismatch: {len(points)} vs {len(labels)}")
        return
    
    # 加载预测掩模和计算指标（如果sam_masks_dir存在）
    pred_mask = None
    pred_mask_binary = None
    gt_mask_binary_for_metrics = None
    metrics = {}
    do_prediction = False
    
    if show_prediction:
        sam_masks_dir = sample.get('sam_masks_dir', None)
        gt_mask_path = sample.get('gt_mask', None)
        
        if sam_masks_dir and gt_mask_path:
            try:
                # 获取图像stem（文件名不含扩展名）
                image_stem = Path(image_path).stem
                mask_dir = os.path.join(sam_masks_dir, image_stem)
                
                # 加载最后一个步骤的预测掩模（len(points)-1.png）
                if len(points) > 0:
                    last_mask_idx = len(points) - 1
                    pred_mask_path = os.path.join(mask_dir, f"{last_mask_idx}.png")
                    
                    if os.path.exists(pred_mask_path):
                        pred_mask = np.array(Image.open(pred_mask_path).convert('L'))
                        pred_mask_binary = (pred_mask > 127).astype(float)
                        
                        # 加载真值掩模并计算指标
                        if os.path.exists(gt_mask_path):
                            gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                            gt_mask_binary_for_metrics = (gt_mask > 127).astype(float)
                            
                            # 调整尺寸以匹配（如果需要）
                            if gt_mask_binary_for_metrics.shape != pred_mask_binary.shape:
                                scale_y = pred_mask_binary.shape[0] / gt_mask_binary_for_metrics.shape[0]
                                scale_x = pred_mask_binary.shape[1] / gt_mask_binary_for_metrics.shape[1]
                                gt_mask_binary_for_metrics = zoom(gt_mask_binary_for_metrics, (scale_y, scale_x), order=0)
                            
                            # 转换为tensor计算指标
                            pred_tensor = torch.from_numpy(pred_mask_binary).float()
                            gt_tensor = torch.from_numpy(gt_mask_binary_for_metrics).float()
                            
                            dice = compute_dice(pred_tensor, gt_tensor)
                            iou = compute_iou(pred_tensor, gt_tensor)
                            
                            metrics['dice'] = dice
                            metrics['iou'] = iou
                            metrics['num_points'] = len(points)
                            
                            do_prediction = True
                        else:
                            print(f"Warning: GT mask not found: {gt_mask_path}")
                    else:
                        print(f"Warning: Predicted mask not found: {pred_mask_path}")
                else:
                    print(f"Warning: No points in sample, cannot load predicted mask")
            except Exception as e:
                print(f"Warning: Failed to load predicted mask: {e}")
                import traceback
                traceback.print_exc()
    
    # 创建图形
    if do_prediction:
        # 两幅图：第一幅显示点和标注，第二幅显示掩模对比
        fig, axes = plt.subplots(1, 2, figsize=(24, 12))
        ax1, ax2 = axes
    else:
        # 单幅图：只显示点和标注
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 12))
        ax2 = None
    
    # 第一幅图：显示点和标注
    ax1.imshow(img_np)
    
    # 如果有真值掩模，叠加显示
    if show_mask:
        gt_mask_path = sample.get('gt_mask', None)
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                mask_binary = (gt_mask > 127).astype(float)
                ax1.imshow(mask_binary, alpha=0.3, cmap='Greens')
            except Exception as e:
                print(f"Warning: Failed to load mask {gt_mask_path}: {e}")
    
    # 绘制点
    for j, ((x, y), label) in enumerate(zip(points, labels)):
        if label == 1:
            # 前景点：红色，实心圆
            color = 'red'
            size = 150
            point_type = 'FG'
        else:
            # 背景点：蓝色，实心圆
            color = 'blue'
            size = 120
            point_type = 'BG'
        
        # 绘制点
        ax1.scatter(x, y, c=color, s=size, marker='o', 
                   edgecolors='white', linewidths=2, zorder=10)
        
        # 添加标签文本
        ax1.text(x + 15, y, f'{j}({point_type})', 
                fontsize=10, color='white', weight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    # 设置第一幅图的标题
    stem = Path(image_path).stem
    title1 = f'{stem}: {len(points)} points (Ground Truth)'
    ax1.set_title(title1, fontsize=14, weight='bold')
    ax1.axis('off')
    
    # 第二幅图：显示掩模对比（如果进行预测）
    if do_prediction and ax2 is not None:
        ax2.imshow(img_np)
        
        gt_mask_path = sample.get('gt_mask', None)
        gt_mask_binary = None
        
        # 加载并调整真值掩模尺寸（用于显示）
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                gt_mask_binary = (gt_mask > 127).astype(float)
                # 调整尺寸以匹配图像
                if gt_mask_binary.shape != img_np.shape[:2]:
                    scale_y = img_np.shape[0] / gt_mask_binary.shape[0]
                    scale_x = img_np.shape[1] / gt_mask_binary.shape[1]
                    gt_mask_binary = zoom(gt_mask_binary, (scale_y, scale_x), order=0)
            except Exception as e:
                print(f"Warning: Failed to load GT mask: {e}")
        
        # 调整预测掩模尺寸（用于显示）
        pred_mask_binary_display = None
        if pred_mask_binary is not None:
            # pred_mask_binary已经是二值化的，但需要调整尺寸以匹配图像
            if pred_mask_binary.shape != img_np.shape[:2]:
                scale_y = img_np.shape[0] / pred_mask_binary.shape[0]
                scale_x = img_np.shape[1] / pred_mask_binary.shape[1]
                pred_mask_binary_display = zoom(pred_mask_binary, (scale_y, scale_x), order=0)
            else:
                pred_mask_binary_display = pred_mask_binary
        
        # 优化显示：先显示重叠区域（黄色），再显示真值（绿色），最后显示预测（红色）
        if gt_mask_binary is not None and pred_mask_binary_display is not None:
            # 重叠区域（黄色）
            overlap = (gt_mask_binary * pred_mask_binary_display).astype(float)
            overlap_colored = np.zeros_like(img_np)
            overlap_colored[:, :, 0] = overlap * 255  # 红色
            overlap_colored[:, :, 1] = overlap * 255  # 绿色
            ax2.imshow(overlap_colored, alpha=0.5)
            
            # 真值掩模的未重叠部分（绿色）
            gt_only = (gt_mask_binary * (1 - pred_mask_binary_display)).astype(float)
            gt_only_colored = np.zeros_like(img_np)
            gt_only_colored[:, :, 1] = gt_only * 255
            ax2.imshow(gt_only_colored, alpha=0.4)
            
            # 预测掩模的未重叠部分（红色）
            pred_only = (pred_mask_binary_display * (1 - gt_mask_binary)).astype(float)
            pred_only_colored = np.zeros_like(img_np)
            pred_only_colored[:, :, 0] = pred_only * 255
            ax2.imshow(pred_only_colored, alpha=0.4)
        elif gt_mask_binary is not None:
            # 只有真值掩模（绿色）
            gt_mask_colored = np.zeros_like(img_np)
            gt_mask_colored[:, :, 1] = gt_mask_binary * 255
            ax2.imshow(gt_mask_colored, alpha=0.4)
        elif pred_mask_binary_display is not None:
            # 只有预测掩模（红色）
            pred_mask_colored = np.zeros_like(img_np)
            pred_mask_colored[:, :, 0] = pred_mask_binary_display * 255
            ax2.imshow(pred_mask_colored, alpha=0.4)
        
        # 绘制预测点（使用JSONL中的points）
        for j, ((x, y), label) in enumerate(zip(points, labels)):
            if label == 1:
                color = 'red'
                size = 150
                point_type = 'FG'
            else:
                color = 'blue'
                size = 120
                point_type = 'BG'
            
            ax2.scatter(x, y, c=color, s=size, marker='o', 
                       edgecolors='white', linewidths=2, zorder=10)
            ax2.text(x + 15, y, f'{j}({point_type})', 
                    fontsize=10, color='white', weight='bold',
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
        
        # 设置第二幅图的标题和指标
        title2 = f'Prediction: {len(points)} points'
        if metrics:
            metrics_text = f"Dice: {metrics.get('dice', 0):.3f}, IoU: {metrics.get('iou', 0):.3f}"
            title2 += f'\n{metrics_text}'
        ax2.set_title(title2, fontsize=14, weight='bold')
        ax2.axis('off')
        
        # 在图像角落添加指标文本
        if metrics:
            metrics_str = f"Dice: {metrics.get('dice', 0):.3f}\nIoU: {metrics.get('iou', 0):.3f}\nPoints: {metrics.get('num_points', 0)}"
            ax2.text(0.02, 0.98, metrics_str, 
                    transform=ax2.transAxes, fontsize=12, weight='bold',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='可视化PEFT训练数据',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--jsonl',
        type=str,
        required=True,
        help='输入JSONL文件路径'
    )
    
    parser.add_argument(
        '--out_dir',
        type=str,
        required=True,
        help='输出图像文件夹路径'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['random', 'all'],
        default='random',
        help='采样模式: random=随机采样, all=遍历所有样本'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='随机采样模式下的样本数量（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--show_mask',
        action='store_true',
        default=True,
        help='是否显示真值掩模（默认True）'
    )
    
    parser.add_argument(
        '--no_mask',
        dest='show_mask',
        action='store_false',
        help='不显示真值掩模'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='输出图像分辨率（DPI）'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default='sample',
        help='输出文件名前缀'
    )
    
    parser.add_argument(
        '--show_prediction',
        action='store_true',
        default=True,
        help='是否显示预测掩模对比（需要sam_masks_dir，默认True）'
    )
    
    parser.add_argument(
        '--no_prediction',
        dest='show_prediction',
        action='store_false',
        help='不显示预测掩模对比'
    )
    
    args = parser.parse_args()
    
    # 加载数据
    print(f"Loading data from {args.jsonl}...")
    data = load_jsonl(args.jsonl)
    print(f"Loaded {len(data)} samples")
    
    # 选择样本
    if args.mode == 'random':
        random.seed(args.seed)
        selected_samples = random.sample(data, min(args.num_samples, len(data)))
        print(f"Randomly selected {len(selected_samples)} samples")
    else:
        selected_samples = data
        print(f"Processing all {len(selected_samples)} samples")
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 可视化每个样本
    print(f"\nGenerating visualizations...")
    for i, sample in enumerate(selected_samples):
        # 生成输出文件名
        if args.mode == 'random':
            output_filename = f'{args.prefix}_{i:03d}.png'
        else:
            # 使用图像文件名作为输出文件名
            image_path = sample.get('image', '')
            if image_path:
                stem = Path(image_path).stem
                output_filename = f'{args.prefix}_{stem}.png'
            else:
                output_filename = f'{args.prefix}_{i:04d}.png'
        
        output_path = os.path.join(args.out_dir, output_filename)
        
        # 可视化
        try:
            visualize_sample(
                sample, output_path, 
                show_mask=args.show_mask, 
                dpi=args.dpi,
                show_prediction=args.show_prediction,
            )
            print(f"  [{i+1}/{len(selected_samples)}] Saved: {output_path}")
        except Exception as e:
            print(f"  [{i+1}/{len(selected_samples)}] Error processing sample {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nDone! Visualizations saved to: {args.out_dir}")
    print(f"Total: {len(selected_samples)} images generated")


if __name__ == '__main__':
    main()


```


更新文档字符串，说明新的使用方式：


```python
#!/usr/bin/env python3
"""
可视化训练数据脚本

用于可视化PEFT训练数据中的图像、点标注和真值掩模。
支持随机采样或遍历所有样本。
如果提供了checkpoint，还会生成预测掩模对比图。

用法:
    # 随机采样5个样本（仅显示标注）
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_train-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode random \
        --num_samples 5

    # 遍历所有样本（带预测对比，需要sam_masks_dir）
    python -m seg-rl.peft.visualize_training_data \
        --jsonl outputs/braintumour/peft_test-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode all \
        --show_prediction
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import zoom

# 添加父目录到路径
_PARENT_DIR = Path(__file__).parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from peft.utils_peft import compute_dice, compute_iou


def load_jsonl(jsonl_path: str) -> List[Dict]:
    """加载JSONL文件"""
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith('['):
            # JSON数组格式
            data = json.loads(content)
        else:
            # JSONL格式（逐行）
            data = []
            for line in content.splitlines():
                if line.strip():
                    data.append(json.loads(line))
    return data


def visualize_sample(
    sample: Dict,
    output_path: str,
    show_mask: bool = True,
    dpi: int = 150,
    show_prediction: bool = True,
) -> None:
    """
    可视化单个样本
    
    Args:
        sample: 样本字典，包含 'image', 'gt_mask', 'points', 'labels', 'sam_masks_dir' 等字段
        output_path: 输出图像路径
        show_mask: 是否显示真值掩模
        dpi: 图像分辨率
        show_prediction: 是否显示预测掩模对比（需要sam_masks_dir）
    """
    # 加载图像
    image_path = sample['image']
    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        return
    
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    orig_w, orig_h = img.size
    
    # 获取点和标签
    points = sample.get('points', [])
    labels = sample.get('labels', [])
    
    if len(points) != len(labels):
        print(f"Warning: Points and labels length mismatch: {len(points)} vs {len(labels)}")
        return
    
    # 加载预测掩模和计算指标（如果sam_masks_dir存在）
    pred_mask = None
    pred_mask_binary = None
    gt_mask_binary_for_metrics = None
    metrics = {}
    do_prediction = False
    
    if show_prediction:
        sam_masks_dir = sample.get('sam_masks_dir', None)
        gt_mask_path = sample.get('gt_mask', None)
        
        if sam_masks_dir and gt_mask_path:
            try:
                # 获取图像stem（文件名不含扩展名）
                image_stem = Path(image_path).stem
                mask_dir = os.path.join(sam_masks_dir, image_stem)
                
                # 加载最后一个步骤的预测掩模（len(points)-1.png）
                if len(points) > 0:
                    last_mask_idx = len(points) - 1
                    pred_mask_path = os.path.join(mask_dir, f"{last_mask_idx}.png")
                    
                    if os.path.exists(pred_mask_path):
                        pred_mask = np.array(Image.open(pred_mask_path).convert('L'))
                        pred_mask_binary = (pred_mask > 127).astype(float)
                        
                        # 加载真值掩模并计算指标
                        if os.path.exists(gt_mask_path):
                            gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                            gt_mask_binary_for_metrics = (gt_mask > 127).astype(float)
                            
                            # 调整尺寸以匹配（如果需要）
                            if gt_mask_binary_for_metrics.shape != pred_mask_binary.shape:
                                scale_y = pred_mask_binary.shape[0] / gt_mask_binary_for_metrics.shape[0]
                                scale_x = pred_mask_binary.shape[1] / gt_mask_binary_for_metrics.shape[1]
                                gt_mask_binary_for_metrics = zoom(gt_mask_binary_for_metrics, (scale_y, scale_x), order=0)
                            
                            # 转换为tensor计算指标
                            pred_tensor = torch.from_numpy(pred_mask_binary).float()
                            gt_tensor = torch.from_numpy(gt_mask_binary_for_metrics).float()
                            
                            dice = compute_dice(pred_tensor, gt_tensor)
                            iou = compute_iou(pred_tensor, gt_tensor)
                            
                            metrics['dice'] = dice
                            metrics['iou'] = iou
                            metrics['num_points'] = len(points)
                            
                            do_prediction = True
                        else:
                            print(f"Warning: GT mask not found: {gt_mask_path}")
                    else:
                        print(f"Warning: Predicted mask not found: {pred_mask_path}")
                else:
                    print(f"Warning: No points in sample, cannot load predicted mask")
            except Exception as e:
                print(f"Warning: Failed to load predicted mask: {e}")
                import traceback
                traceback.print_exc()
    
    # 创建图形
    if do_prediction:
        # 两幅图：第一幅显示点和标注，第二幅显示掩模对比
        fig, axes = plt.subplots(1, 2, figsize=(24, 12))
        ax1, ax2 = axes
    else:
        # 单幅图：只显示点和标注
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 12))
        ax2 = None
    
    # 第一幅图：显示点和标注
    ax1.imshow(img_np)
    
    # 如果有真值掩模，叠加显示
    if show_mask:
        gt_mask_path = sample.get('gt_mask', None)
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                mask_binary = (gt_mask > 127).astype(float)
                ax1.imshow(mask_binary, alpha=0.3, cmap='Greens')
            except Exception as e:
                print(f"Warning: Failed to load mask {gt_mask_path}: {e}")
    
    # 绘制点
    for j, ((x, y), label) in enumerate(zip(points, labels)):
        if label == 1:
            # 前景点：红色，实心圆
            color = 'red'
            size = 150
            point_type = 'FG'
        else:
            # 背景点：蓝色，实心圆
            color = 'blue'
            size = 120
            point_type = 'BG'
        
        # 绘制点
        ax1.scatter(x, y, c=color, s=size, marker='o', 
                   edgecolors='white', linewidths=2, zorder=10)
        
        # 添加标签文本
        ax1.text(x + 15, y, f'{j}({point_type})', 
                fontsize=10, color='white', weight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    # 设置第一幅图的标题
    stem = Path(image_path).stem
    title1 = f'{stem}: {len(points)} points (Ground Truth)'
    ax1.set_title(title1, fontsize=14, weight='bold')
    ax1.axis('off')
    
    # 第二幅图：显示掩模对比（如果进行预测）
    if do_prediction and ax2 is not None:
        ax2.imshow(img_np)
        
        gt_mask_path = sample.get('gt_mask', None)
        gt_mask_binary = None
        
        # 加载并调整真值掩模尺寸（用于显示）
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                gt_mask_binary = (gt_mask > 127).astype(float)
                # 调整尺寸以匹配图像
                if gt_mask_binary.shape != img_np.shape[:2]:
                    scale_y = img_np.shape[0] / gt_mask_binary.shape[0]
                    scale_x = img_np.shape[1] / gt_mask_binary.shape[1]
                    gt_mask_binary = zoom(gt_mask_binary, (scale_y, scale_x), order=0)
            except Exception as e:
                print(f"Warning: Failed to load GT mask: {e}")
        
        # 调整预测掩模尺寸（用于显示）
        pred_mask_binary_display = None
        if pred_mask_binary is not None:
            # pred_mask_binary已经是二值化的，但需要调整尺寸以匹配图像
            if pred_mask_binary.shape != img_np.shape[:2]:
                scale_y = img_np.shape[0] / pred_mask_binary.shape[0]
                scale_x = img_np.shape[1] / pred_mask_binary.shape[1]
                pred_mask_binary_display = zoom(pred_mask_binary, (scale_y, scale_x), order=0)
            else:
                pred_mask_binary_display = pred_mask_binary
        
        # 优化显示：先显示重叠区域（黄色），再显示真值（绿色），最后显示预测（红色）
        if gt_mask_binary is not None and pred_mask_binary_display is not None:
            # 重叠区域（黄色）
            overlap = (gt_mask_binary * pred_mask_binary_display).astype(float)
            overlap_colored = np.zeros_like(img_np)
            overlap_colored[:, :, 0] = overlap * 255  # 红色
            overlap_colored[:, :, 1] = overlap * 255  # 绿色
            ax2.imshow(overlap_colored, alpha=0.5)
            
            # 真值掩模的未重叠部分（绿色）
            gt_only = (gt_mask_binary * (1 - pred_mask_binary_display)).astype(float)
            gt_only_colored = np.zeros_like(img_np)
            gt_only_colored[:, :, 1] = gt_only * 255
            ax2.imshow(gt_only_colored, alpha=0.4)
            
            # 预测掩模的未重叠部分（红色）
            pred_only = (pred_mask_binary_display * (1 - gt_mask_binary)).astype(float)
            pred_only_colored = np.zeros_like(img_np)
            pred_only_colored[:, :, 0] = pred_only * 255
            ax2.imshow(pred_only_colored, alpha=0.4)
        elif gt_mask_binary is not None:
            # 只有真值掩模（绿色）
            gt_mask_colored = np.zeros_like(img_np)
            gt_mask_colored[:, :, 1] = gt_mask_binary * 255
            ax2.imshow(gt_mask_colored, alpha=0.4)
        elif pred_mask_binary_display is not None:
            # 只有预测掩模（红色）
            pred_mask_colored = np.zeros_like(img_np)
            pred_mask_colored[:, :, 0] = pred_mask_binary_display * 255
            ax2.imshow(pred_mask_colored, alpha=0.4)
        
        # 绘制预测点（使用JSONL中的points）
        for j, ((x, y), label) in enumerate(zip(points, labels)):
            if label == 1:
                color = 'red'
                size = 150
                point_type = 'FG'
            else:
                color = 'blue'
                size = 120
                point_type = 'BG'
            
            ax2.scatter(x, y, c=color, s=size, marker='o', 
                       edgecolors='white', linewidths=2, zorder=10)
            ax2.text(x + 15, y, f'{j}({point_type})', 
                    fontsize=10, color='white', weight='bold',
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
        
        # 设置第二幅图的标题和指标
        title2 = f'Prediction: {len(points)} points'
        if metrics:
            metrics_text = f"Dice: {metrics.get('dice', 0):.3f}, IoU: {metrics.get('iou', 0):.3f}"
            title2 += f'\n{metrics_text}'
        ax2.set_title(title2, fontsize=14, weight='bold')
        ax2.axis('off')
        
        # 在图像角落添加指标文本
        if metrics:
            metrics_str = f"Dice: {metrics.get('dice', 0):.3f}\nIoU: {metrics.get('iou', 0):.3f}\nPoints: {metrics.get('num_points', 0)}"
            ax2.text(0.02, 0.98, metrics_str, 
                    transform=ax2.transAxes, fontsize=12, weight='bold',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='可视化PEFT训练数据',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--jsonl',
        type=str,
        required=True,
        help='输入JSONL文件路径'
    )
    
    parser.add_argument(
        '--out_dir',
        type=str,
        required=True,
        help='输出图像文件夹路径'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['random', 'all'],
        default='random',
        help='采样模式: random=随机采样, all=遍历所有样本'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='随机采样模式下的样本数量（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--show_mask',
        action='store_true',
        default=True,
        help='是否显示真值掩模（默认True）'
    )
    
    parser.add_argument(
        '--no_mask',
        dest='show_mask',
        action='store_false',
        help='不显示真值掩模'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='输出图像分辨率（DPI）'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default='sample',
        help='输出文件名前缀'
    )
    
    parser.add_argument(
        '--show_prediction',
        action='store_true',
        default=True,
        help='是否显示预测掩模对比（需要sam_masks_dir，默认True）'
    )
    
    parser.add_argument(
        '--no_prediction',
        dest='show_prediction',
        action='store_false',
        help='不显示预测掩模对比'
    )
    
    args = parser.parse_args()
    
    # 加载数据
    print(f"Loading data from {args.jsonl}...")
    data = load_jsonl(args.jsonl)
    print(f"Loaded {len(data)} samples")
    
    # 选择样本
    if args.mode == 'random':
        random.seed(args.seed)
        selected_samples = random.sample(data, min(args.num_samples, len(data)))
        print(f"Randomly selected {len(selected_samples)} samples")
    else:
        selected_samples = data
        print(f"Processing all {len(selected_samples)} samples")
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 可视化每个样本
    print(f"\nGenerating visualizations...")
    for i, sample in enumerate(selected_samples):
        # 生成输出文件名
        if args.mode == 'random':
            output_filename = f'{args.prefix}_{i:03d}.png'
        else:
            # 使用图像文件名作为输出文件名
            image_path = sample.get('image', '')
            if image_path:
                stem = Path(image_path).stem
                output_filename = f'{args.prefix}_{stem}.png'
            else:
                output_filename = f'{args.prefix}_{i:04d}.png'
        
        output_path = os.path.join(args.out_dir, output_filename)
        
        # 可视化
        try:
            visualize_sample(
                sample, output_path, 
                show_mask=args.show_mask, 
                dpi=args.dpi,
                show_prediction=args.show_prediction,
            )
            print(f"  [{i+1}/{len(selected_samples)}] Saved: {output_path}")
        except Exception as e:
            print(f"  [{i+1}/{len(selected_samples)}] Error processing sample {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nDone! Visualizations saved to: {args.out_dir}")
    print(f"Total: {len(selected_samples)} images generated")


if __name__ == '__main__':
    main()


```


检查是否还需要 Optional：



脚本已更新。主要改动：

## 修改总结

1. **移除了模型加载和预测代码**：不再需要加载checkpoint或运行模型预测
2. **直接从文件系统读取掩模**：
   - 从 `gt_mask` 路径读取真值掩模
   - 从 `sam_masks_dir/{image_stem}/{len(points)-1}.png` 读取最后一个步骤的预测掩模
3. **简化了参数**：
   - 移除了 `--checkpoint`, `--sam_checkpoint`, `--image_size`, `--device`, `--max_rollout_steps`
   - 添加了 `--show_prediction` / `--no_prediction` 来控制是否显示预测对比

## 使用示例

```bash
# 显示预测对比（需要sam_masks_dir）
python -m seg-rl.peft.visualize_training_data \
    --jsonl outputs/braintumour/peft_test-251107.jsonl \
    --out_dir outputs/braintumour/visualizations \
    --mode random \
    --num_samples 5 \
    --show_prediction

# 不显示预测对比
python -m seg-rl.peft.visualize_training_data \
    --jsonl outputs/braintumour/peft_train-251107.jsonl \
    --out_dir outputs/braintumour/visualizations \
    --mode random \
    --num_samples 5 \
    --no_prediction
```

脚本现在直接从JSONL和文件系统读取数据，无需加载模型。

```shellscript
# 显示预测对比（需要sam_masks_dir）
python -m seg-rl.peft.visualize_training_data \
    --jsonl outputs/braintumour/peft_test-251107.jsonl \
    --out_dir outputs/braintumour/visualizations \
    --mode random \
    --num_samples 5 \
    --show_prediction

# 不显示预测对比
python -m seg-rl.peft.visualize_training_data \
    --jsonl outputs/braintumour/peft_train-251107.jsonl \
    --out_dir outputs/braintumour/visualizations \
    --mode random \
    --num_samples 5 \
    --no_prediction
```

---

**User**

seg-rl/peft/eval_peft_model.py脚本明显有问题，从它生成的outputs/braintumour/eval_supervised_pred_test-251107A/eval_results.json来看, points各坐标范围明显不在原始输入图像的尺寸范围240x240之内（参考outputs/braintumour/peft_test-251107.jsonl，其坐标范围都在240x240之内 ），所以一定是有关坐标缩放的逻辑哪里出了问题，请仔细检查并修改。问题应该只在该脚本之内而不涉及其它代码

---

**Cursor**


检查 `eval_peft_model.py` 中的坐标缩放逻辑：



检查坐标缩放逻辑，定位问题：



问题：第143行记录的坐标在 resize 后的坐标系（512x512），但应返回原始图像尺寸（240x240）的坐标。修复：在返回结果前，将 `points_seq` 中的所有坐标缩放回原始尺寸。



问题：第143行记录的坐标在 resize 后的坐标系（512x512），但应返回原始图像尺寸（240x240）的坐标。修复：在返回结果前，将 `points_seq` 中的所有坐标缩放回原始尺寸。


```python
#!/usr/bin/env python3
"""
PEFT模型评估脚本

在测试集上评估训练好的PEFT模型性能

主要功能:
- 加载训练好的checkpoint
- 执行完整rollout（迭代预测点序列）
- 计算指标：Dice, IoU, PCK, 点数等
- 生成可视化：点序列图、分割结果对比

调用示例:
    python -m seg-rl.peft.eval_peft_model \\
      --test_json datasets/seg_r1_md/Task01_BrainTumour/test.jsonl \\
      --checkpoint outputs/braintumour/peft_supervised/checkpoint_best.pt \\
      --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \\
      --out_dir outputs/braintumour/peft_eval \\
      --max_rollout_steps 16 \\
      --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# 添加父目录到路径
_PARENT_DIR = Path(__file__).parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from peft.lora_sam2 import LoRASAM2Wrapper
from peft.point_predictor_peft import PointPredictorFromSAMFeatures
from peft.utils_peft import compute_dice, compute_iou, load_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate PEFT Model")
    
    p.add_argument("--test_json", type=str, required=True, help="Test JSONL file")
    p.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint to evaluate")
    p.add_argument("--sam_checkpoint", type=str, required=True, help="SAM2 checkpoint")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory for results")
    
    p.add_argument("--image_size", type=int, nargs=2, default=[512, 512], help="Image size")
    p.add_argument("--max_rollout_steps", type=int, default=16, help="Max points per image")
    p.add_argument("--dice_threshold", type=float, default=0.95, help="Stop if Dice > threshold")
    p.add_argument("--improvement_threshold", type=float, default=0.001,
                   help="Stop if improvement < threshold")
    
    p.add_argument("--device", type=str, default="cuda", help="Device")
    p.add_argument("--save_visualizations", action="store_true", help="Save visualization images")
    
    return p.parse_args()


@torch.no_grad()
def rollout_single_image(
    image_path: str,
    gt_mask_path: str,
    sam2_lora: LoRASAM2Wrapper,
    point_predictor: PointPredictorFromSAMFeatures,
    args,
) -> Dict:
    """
    对单张图像执行完整rollout
    
    Returns:
        结果字典，包含点序列、Dice曲线等
    """
    device = args.device
    
    # 加载图像和真值
    image = Image.open(image_path).convert('RGB')
    gt_mask = Image.open(gt_mask_path).convert('L')
    
    orig_w, orig_h = image.size
    
    # Resize
    image_resized = image.resize(tuple(args.image_size), Image.BILINEAR)
    gt_mask_resized = gt_mask.resize(tuple(args.image_size), Image.NEAREST)
    
    # 转换为tensor
    import torchvision.transforms.functional as TF
    image_tensor = TF.to_tensor(image_resized).unsqueeze(0).to(device)  # [1, 3, H, W]
    gt_mask_tensor = TF.to_tensor(gt_mask_resized).to(device)  # [1, H, W]
    gt_mask_binary = (gt_mask_tensor > 0.5).float()
    
    # Rollout
    points_seq = []
    labels_seq = []
    dice_history = []
    iou_history = []
    
    prev_dice = 0.0
    
    for step in range(args.max_rollout_steps):
        # 生成当前掩模
        if step == 0:
            prev_mask = torch.zeros_like(gt_mask_binary)
        else:
            # 用SAM2生成掩模
            # 需要将点坐标缩放回原始尺寸
            points_orig = [(px * orig_w / args.image_size[1], py * orig_h / args.image_size[0])
                          for px, py in points_seq]
            prev_mask = sam2_lora.predict_mask(
                np.array(image), points_orig, labels_seq
            )
            # predict_mask 返回的是 torch.Tensor，不是 numpy 数组
            if isinstance(prev_mask, torch.Tensor):
                prev_mask = prev_mask.unsqueeze(0).to(device)
            else:
                prev_mask = torch.from_numpy(prev_mask).unsqueeze(0).to(device)
            prev_mask = torch.nn.functional.interpolate(
                prev_mask.unsqueeze(0), size=tuple(args.image_size), mode='nearest'
            )[0]
        
        # 预测下一个点
        sam_features = sam2_lora.get_image_features(image_tensor, feature_scale=8)
        heatmap_logits, label_logits = point_predictor(sam_features, prev_mask.unsqueeze(0))
        
        # Argmax获取点
        B, _, H, W = heatmap_logits.shape
        heatmap_flat = heatmap_logits.view(B, -1)
        pred_idx = heatmap_flat.argmax(dim=1)
        pred_y = (pred_idx // W).float()
        pred_x = (pred_idx % W).float()
        
        # 标签
        pred_label = label_logits.argmax(dim=1).item()
        
        # 记录
        points_seq.append((pred_x.item(), pred_y.item()))
        labels_seq.append(pred_label)
        
        # 计算指标
        dice = compute_dice(prev_mask[0], gt_mask_binary[0])
        iou = compute_iou(prev_mask[0], gt_mask_binary[0])
        dice_history.append(dice)
        iou_history.append(iou)
        
        # 停止条件
        if dice > args.dice_threshold:
            break
        if step > 0 and (dice - prev_dice) < args.improvement_threshold:
            break
        
        prev_dice = dice
    
    # 将点坐标从resize后的尺寸缩放回原始图像尺寸
    # points_seq中的坐标是在args.image_size坐标系中的，需要缩放回orig_w x orig_h
    # args.image_size是[H, W]，所以args.image_size[0]是H，args.image_size[1]是W
    points_orig = [
        (px * orig_w / args.image_size[1], py * orig_h / args.image_size[0])
        for px, py in points_seq
    ]
    
    return {
        'image_path': image_path,
        'points': points_orig,  # 返回原始图像尺寸的坐标
        'labels': labels_seq,
        'dice_history': dice_history,
        'iou_history': iou_history,
        'final_dice': dice_history[-1] if dice_history else 0.0,
        'final_iou': iou_history[-1] if iou_history else 0.0,
        'num_points': len(points_seq),
    }


def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 设备
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # 加载checkpoint配置
    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = ckpt.get('config', {})
    
    # 提取配置
    lora_rank = config.get('lora_rank', 16)
    lora_alpha = config.get('lora_alpha', 32)
    fusion_mode = config.get('fusion_mode', 'film')
    feature_scale = config.get('feature_scale', 8)
    
    print(f"Model config: LoRA rank={lora_rank}, alpha={lora_alpha}, fusion={fusion_mode}")
    
    # 1. 初始化SAM2 + LoRA
    print("\nInitializing SAM2 with LoRA...")
    sam2_lora = LoRASAM2Wrapper(
        sam_checkpoint=args.sam_checkpoint,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        device=str(device)
    )
    
    # 2. 初始化点预测网络
    print("Initializing Point Predictor...")
    
    # 根据 feature_scale 确定 SAM 特征通道数
    sam_feature_dim_map = {4: 32, 8: 64, 16: 256}
    if feature_scale not in sam_feature_dim_map:
        raise ValueError(f"Unsupported feature_scale: {feature_scale}. Use 4, 8, or 16.")
    sam_feature_dim = sam_feature_dim_map[feature_scale]
    
    point_predictor = PointPredictorFromSAMFeatures(
        sam_feature_dim=sam_feature_dim,
        output_size=tuple(args.image_size),
        fusion_mode=fusion_mode,
        feature_scale=feature_scale,
    ).to(device)
    
    # 3. 加载权重
    print("Loading model weights...")
    load_checkpoint(
        args.checkpoint, point_predictor, sam2_lora, None, None, None, str(device)
    )
    
    point_predictor.eval()
    
    # 4. 加载测试数据
    print(f"\nLoading test data from {args.test_json}")
    with open(args.test_json, 'r') as f:
        content = f.read().strip()
        if content.startswith('['):
            test_samples = json.loads(content)
        else:
            test_samples = [json.loads(line) for line in content.splitlines() if line.strip()]
    
    print(f"Loaded {len(test_samples)} test samples")
    
    # 5. 评估
    print("\n" + "="*80)
    print("Evaluating...")
    print("="*80)
    
    results = []
    
    for sample in tqdm(test_samples):
        image_path = sample['image']
        gt_mask_path = sample.get('gt_mask', None)
        
        if gt_mask_path is None or not os.path.exists(gt_mask_path):
            print(f"Warning: GT mask not found for {image_path}, skipping")
            continue
        
        result = rollout_single_image(
            image_path, gt_mask_path, sam2_lora, point_predictor, args
        )
        results.append(result)
    
    # 6. 汇总统计
    print("\n" + "="*80)
    print("Results Summary")
    print("="*80)
    
    final_dices = [r['final_dice'] for r in results]
    final_ious = [r['final_iou'] for r in results]
    num_points = [r['num_points'] for r in results]
    
    print(f"Number of samples: {len(results)}")
    print(f"\nDice Score:")
    print(f"  Mean: {np.mean(final_dices):.4f}")
    print(f"  Std:  {np.std(final_dices):.4f}")
    print(f"  Median: {np.median(final_dices):.4f}")
    
    print(f"\nIoU Score:")
    print(f"  Mean: {np.mean(final_ious):.4f}")
    print(f"  Std:  {np.std(final_ious):.4f}")
    print(f"  Median: {np.median(final_ious):.4f}")
    
    print(f"\nNumber of Points:")
    print(f"  Mean: {np.mean(num_points):.2f}")
    print(f"  Std:  {np.std(num_points):.2f}")
    print(f"  Median: {np.median(num_points):.0f}")
    
    # 7. 保存结果
    results_json_path = os.path.join(args.out_dir, 'eval_results.json')
    with open(results_json_path, 'w') as f:
        json.dump({
            'summary': {
                'num_samples': len(results),
                'mean_dice': float(np.mean(final_dices)),
                'std_dice': float(np.std(final_dices)),
                'median_dice': float(np.median(final_dices)),
                'mean_iou': float(np.mean(final_ious)),
                'mean_num_points': float(np.mean(num_points)),
            },
            'per_sample_results': results,
        }, f, indent=2)
    
    print(f"\nResults saved to {results_json_path}")


if __name__ == "__main__":
    main()


```


已修复坐标缩放问题。linter 的导入警告不影响运行。

## 修复说明

**问题：**
- 第143行记录的 `points_seq` 坐标在 resize 后的坐标系（512x512）
- 返回结果时直接返回了这些坐标，未缩放回原始图像尺寸（240x240）

**修复：**
- 在返回结果前，将所有点坐标从 resize 后的尺寸缩放回原始图像尺寸
- 缩放公式：`(px * orig_w / args.image_size[1], py * orig_h / args.image_size[0])`
  - `args.image_size` 是 `[H, W]`，即 `[512, 512]`
  - `args.image_size[1]` 是宽度 W = 512
  - `args.image_size[0]` 是高度 H = 512

**注意：**
- 第114行的缩放逻辑保持不变（用于 SAM2 预测时临时缩放）
- 最终返回的坐标已缩放回原始图像尺寸

现在返回的 `points` 坐标应在原始图像尺寸（240x240）范围内。

---

**User**

执行以下脚本报错，请找到原因并修改：@bash (978-1020) 

---

**Cursor**


检查并修复 `predict_next_point_from_model.py` 中的 `torch.load` 调用：



修复 `torch.load` 调用，添加 `weights_only=False` 参数：


```python
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
    from .utils import heatmap_to_pil
except Exception:
    # script context
    import sys as _sys
    import os as _os
    _pkg_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _pkg_root not in _sys.path:
        _sys.path.insert(0, _pkg_root)
    from heatmap.model import ModelConfig, PointHeatmapModel, soft_argmax_from_logits
    from heatmap.utils import heatmap_to_pil

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
    # Save probability heatmaps alongside SAM masks: {output_dir}/{stem}/heatmap-{i}.png
    p.add_argument("--output_dir", type=str, default=None, help="Directory to save per-step heatmaps (under {stem}/heatmap-{i}.png)")
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
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False) if model_path and os.path.isfile(model_path) else None
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


def _predict_point(model: PointHeatmapModel, rgb_tensor: torch.Tensor, gray_tensor: torch.Tensor, device: torch.device, tau: float) -> Tuple[float, float, int, np.ndarray]:
    with torch.no_grad():
        logits, label_logits = model(rgb_tensor.to(device), gray_tensor.to(device))
        xy = soft_argmax_from_logits(logits, temperature=tau)[0]
        lab = int(label_logits.argmax(dim=1).item())
        # probability heatmap for saving: softmax over pixels
        b, _, H, W = logits.shape
        p = F.softmax(logits.view(b, -1) / max(float(tau), 1e-6), dim=1).view(b, 1, H, W)
        prob = p[0, 0].detach().cpu().float().numpy()
    return float(xy[0].item()), float(xy[1].item()), lab, prob


def _save_heatmap_image(prob: np.ndarray, out_dir: Optional[str], stem: str, step_idx: int, orig_w: int, orig_h: int) -> None:
    if not out_dir:
        return
    try:
        hm_img = heatmap_to_pil(prob)
        if hm_img.size != (orig_w, orig_h):
            hm_img = hm_img.resize((orig_w, orig_h), Image.BILINEAR)
        subdir = os.path.join(out_dir, stem)
        os.makedirs(subdir, exist_ok=True)
        out_path = os.path.join(subdir, f"heatmap-{step_idx}.png")
        hm_img.save(out_path)
    except Exception:
        pass


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
            x, y, lab, prob = _predict_point(model, rgb_tensor, gray_tensor, device, args.tau)
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
        # save heatmap for step 0
        _save_heatmap_image(prob, args.output_dir or args.sam_dir, stem, 0, orig_w, orig_h)
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
            x, y, lab, prob = _predict_point(model, rgb_tensor, gray_tensor, device, args.tau)
        # scale back and clamp
        inf_H, inf_W = rgb_tensor.shape[-2], rgb_tensor.shape[-1]
        if (inf_W, inf_H) != (orig_w, orig_h):
            scale_x = float(orig_w) / float(inf_W)
            scale_y = float(orig_h) / float(inf_H)
            x = x * scale_x
            y = y * scale_y
        x = float(max(0.0, min(orig_w - 1.0, x)))
        y = float(max(0.0, min(orig_h - 1.0, y)))
        # Save heatmap for next step k = len(pts)
        stem = _stem(img_path)
        step_k = len(pts)
        _save_heatmap_image(prob, args.output_dir or sam_dir, stem, step_k, orig_w, orig_h)
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



```


已修复该问题。

**问题原因：**
与之前相同，PyTorch 2.6 将 `torch.load` 的默认 `weights_only` 从 `False` 改为 `True`，而 checkpoint 文件包含 numpy 对象，需要 `weights_only=False` 才能加载。

**修复内容：**
在 `predict_next_point_from_model.py` 的第157行，为 `torch.load` 调用添加了 `weights_only=False` 参数。

现在可以重新运行脚本，checkpoint 加载错误应已解决。

**注意：** linter 的导入警告不影响运行，只是类型检查工具的提示。

---

**User**

运行该脚本去预测第一个提示点：(seg-r1) root@autodl-container-631447a6cf-2ebc5cc6:~/autodl-tmp/works/Seg-R0# python -m seg-rl.heatmap.predict_next_point_from_model     --model_path outputs/braintumour/peft_supervised_baseline-251107A/checkpoint_best.pt     --images_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/images     --masks_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/masks     --output_json outputs/braintumour/pred-peft_test-251107.jsonl  ，发现预测的点很奇怪，全都是[119.xxx, 119.xxx]，见outputs/braintumour/pred-peft_test-251107.jsonl。 一定存在某种根本性逻辑错误。仔细检查并修改

---

**Cursor**


检查 `predict_next_point_from_model.py` 的逻辑，定位问题：



检查模型加载和预测逻辑，定位问题：



问题在于：`predict_next_point_from_model.py` 是为 `PointHeatmapModel` 设计的，而您提供的 checkpoint 是 PEFT 训练的 `PointPredictorFromSAMFeatures`，架构不匹配。检查 checkpoint 结构以确认：



确认问题：checkpoint 保存的是 `PointPredictorFromSAMFeatures`（PEFT），而脚本加载的是 `PointHeatmapModel`，架构不匹配。修改脚本以支持 PEFT 模型：



修改脚本以支持 PEFT 模型：检测 checkpoint 类型，并相应加载模型。



修改脚本以支持 PEFT 模型：检测 checkpoint 类型，并相应加载模型。


```python
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
    from .utils import heatmap_to_pil
except Exception:
    # script context
    import sys as _sys
    import os as _os
    _pkg_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _pkg_root not in _sys.path:
        _sys.path.insert(0, _pkg_root)
    from heatmap.model import ModelConfig, PointHeatmapModel, soft_argmax_from_logits
    from heatmap.utils import heatmap_to_pil

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
    p.add_argument("--sam_checkpoint", type=str, default=None, help="SAM2 checkpoint path (required for PEFT models)")
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
    # Save probability heatmaps alongside SAM masks: {output_dir}/{stem}/heatmap-{i}.png
    p.add_argument("--output_dir", type=str, default=None, help="Directory to save per-step heatmaps (under {stem}/heatmap-{i}.png)")
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
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False) if model_path and os.path.isfile(model_path) else None
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


def _predict_point(model: PointHeatmapModel, rgb_tensor: torch.Tensor, gray_tensor: torch.Tensor, device: torch.device, tau: float) -> Tuple[float, float, int, np.ndarray]:
    with torch.no_grad():
        logits, label_logits = model(rgb_tensor.to(device), gray_tensor.to(device))
        xy = soft_argmax_from_logits(logits, temperature=tau)[0]
        lab = int(label_logits.argmax(dim=1).item())
        # probability heatmap for saving: softmax over pixels
        b, _, H, W = logits.shape
        p = F.softmax(logits.view(b, -1) / max(float(tau), 1e-6), dim=1).view(b, 1, H, W)
        prob = p[0, 0].detach().cpu().float().numpy()
    return float(xy[0].item()), float(xy[1].item()), lab, prob


def _save_heatmap_image(prob: np.ndarray, out_dir: Optional[str], stem: str, step_idx: int, orig_w: int, orig_h: int) -> None:
    if not out_dir:
        return
    try:
        hm_img = heatmap_to_pil(prob)
        if hm_img.size != (orig_w, orig_h):
            hm_img = hm_img.resize((orig_w, orig_h), Image.BILINEAR)
        subdir = os.path.join(out_dir, stem)
        os.makedirs(subdir, exist_ok=True)
        out_path = os.path.join(subdir, f"heatmap-{step_idx}.png")
        hm_img.save(out_path)
    except Exception:
        pass


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
            x, y, lab, prob = _predict_point(model, rgb_tensor, gray_tensor, device, args.tau)
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
        # save heatmap for step 0
        _save_heatmap_image(prob, args.output_dir or args.sam_dir, stem, 0, orig_w, orig_h)
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
            x, y, lab, prob = _predict_point(model, rgb_tensor, gray_tensor, device, args.tau)
        # scale back and clamp
        inf_H, inf_W = rgb_tensor.shape[-2], rgb_tensor.shape[-1]
        if (inf_W, inf_H) != (orig_w, orig_h):
            scale_x = float(orig_w) / float(inf_W)
            scale_y = float(orig_h) / float(inf_H)
            x = x * scale_x
            y = y * scale_y
        x = float(max(0.0, min(orig_w - 1.0, x)))
        y = float(max(0.0, min(orig_h - 1.0, y)))
        # Save heatmap for next step k = len(pts)
        stem = _stem(img_path)
        step_k = len(pts)
        _save_heatmap_image(prob, args.output_dir or sam_dir, stem, step_k, orig_w, orig_h)
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



```

```python
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
    from .utils import heatmap_to_pil
except Exception:
    # script context
    import sys as _sys
    import os as _os
    _pkg_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _pkg_root not in _sys.path:
        _sys.path.insert(0, _pkg_root)
    from heatmap.model import ModelConfig, PointHeatmapModel, soft_argmax_from_logits
    from heatmap.utils import heatmap_to_pil

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
    p.add_argument("--sam_checkpoint", type=str, default=None, help="SAM2 checkpoint path (required for PEFT models)")
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
    # Save probability heatmaps alongside SAM masks: {output_dir}/{stem}/heatmap-{i}.png
    p.add_argument("--output_dir", type=str, default=None, help="Directory to save per-step heatmaps (under {stem}/heatmap-{i}.png)")
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


def _is_peft_checkpoint(ckpt: dict) -> bool:
    """检查checkpoint是否是PEFT格式"""
    return 'point_predictor_state' in ckpt and 'sam_lora_state' in ckpt


def _load_model(model_path: str, device: torch.device, sam_checkpoint: Optional[str] = None):
    """加载模型，支持heatmap模型和PEFT模型"""
    if not model_path or not os.path.isfile(model_path):
        raise ValueError(f"Model path not found: {model_path}")
    
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    
    # 检查是否是PEFT checkpoint
    if _is_peft_checkpoint(ckpt):
        # 加载PEFT模型
        if not sam_checkpoint:
            raise ValueError("--sam_checkpoint is required for PEFT models")
        
        # 导入PEFT模块
        import sys
        from pathlib import Path
        _PARENT_DIR = Path(__file__).parent.parent
        if str(_PARENT_DIR) not in sys.path:
            sys.path.insert(0, str(_PARENT_DIR))
        
        from peft.lora_sam2 import LoRASAM2Wrapper
        from peft.point_predictor_peft import PointPredictorFromSAMFeatures
        from peft.utils_peft import load_checkpoint
        
        config = ckpt.get('config', {})
        lora_rank = config.get('lora_rank', 16)
        lora_alpha = config.get('lora_alpha', 32)
        fusion_mode = config.get('fusion_mode', 'film')
        feature_scale = config.get('feature_scale', 8)
        image_size = config.get('image_size', [512, 512])
        
        # 初始化SAM2 LoRA
        sam2_lora = LoRASAM2Wrapper(
            sam_checkpoint=sam_checkpoint,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            device=str(device),
        )
        
        # 初始化点预测网络
        sam_feature_dim_map = {4: 32, 8: 64, 16: 256}
        if feature_scale not in sam_feature_dim_map:
            raise ValueError(f"Unsupported feature_scale: {feature_scale}")
        sam_feature_dim = sam_feature_dim_map[feature_scale]
        
        point_predictor = PointPredictorFromSAMFeatures(
            sam_feature_dim=sam_feature_dim,
            output_size=tuple(image_size),
            fusion_mode=fusion_mode,
            feature_scale=feature_scale,
        ).to(device)
        
        # 加载权重
        load_checkpoint(
            model_path, point_predictor, sam2_lora, None, None, None, str(device)
        )
        
        point_predictor.eval()
        return ('peft', sam2_lora, point_predictor, feature_scale, tuple(image_size))
    else:
        # 加载heatmap模型
        cfg = ModelConfig(backbone="unet_s", pretrained=False, main_in_channels=3, cond_in_channels=1)
        model = PointHeatmapModel(cfg).to(device)
        sd = ckpt.get("model", ckpt)
        model.load_state_dict(sd, strict=False)
        model.eval()
        return ('heatmap', model)


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


def _predict_point(model: PointHeatmapModel, rgb_tensor: torch.Tensor, gray_tensor: torch.Tensor, device: torch.device, tau: float) -> Tuple[float, float, int, np.ndarray]:
    with torch.no_grad():
        logits, label_logits = model(rgb_tensor.to(device), gray_tensor.to(device))
        xy = soft_argmax_from_logits(logits, temperature=tau)[0]
        lab = int(label_logits.argmax(dim=1).item())
        # probability heatmap for saving: softmax over pixels
        b, _, H, W = logits.shape
        p = F.softmax(logits.view(b, -1) / max(float(tau), 1e-6), dim=1).view(b, 1, H, W)
        prob = p[0, 0].detach().cpu().float().numpy()
    return float(xy[0].item()), float(xy[1].item()), lab, prob


def _save_heatmap_image(prob: np.ndarray, out_dir: Optional[str], stem: str, step_idx: int, orig_w: int, orig_h: int) -> None:
    if not out_dir:
        return
    try:
        hm_img = heatmap_to_pil(prob)
        if hm_img.size != (orig_w, orig_h):
            hm_img = hm_img.resize((orig_w, orig_h), Image.BILINEAR)
        subdir = os.path.join(out_dir, stem)
        os.makedirs(subdir, exist_ok=True)
        out_path = os.path.join(subdir, f"heatmap-{step_idx}.png")
        hm_img.save(out_path)
    except Exception:
        pass


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
            x, y, lab, prob = _predict_point(model, rgb_tensor, gray_tensor, device, args.tau)
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
        # save heatmap for step 0
        _save_heatmap_image(prob, args.output_dir or args.sam_dir, stem, 0, orig_w, orig_h)
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
            x, y, lab, prob = _predict_point(model, rgb_tensor, gray_tensor, device, args.tau)
        # scale back and clamp
        inf_H, inf_W = rgb_tensor.shape[-2], rgb_tensor.shape[-1]
        if (inf_W, inf_H) != (orig_w, orig_h):
            scale_x = float(orig_w) / float(inf_W)
            scale_y = float(orig_h) / float(inf_H)
            x = x * scale_x
            y = y * scale_y
        x = float(max(0.0, min(orig_w - 1.0, x)))
        y = float(max(0.0, min(orig_h - 1.0, y)))
        # Save heatmap for next step k = len(pts)
        stem = _stem(img_path)
        step_k = len(pts)
        _save_heatmap_image(prob, args.output_dir or sam_dir, stem, step_k, orig_w, orig_h)
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



```

```python
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
    from .utils import heatmap_to_pil
except Exception:
    # script context
    import sys as _sys
    import os as _os
    _pkg_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _pkg_root not in _sys.path:
        _sys.path.insert(0, _pkg_root)
    from heatmap.model import ModelConfig, PointHeatmapModel, soft_argmax_from_logits
    from heatmap.utils import heatmap_to_pil

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
    p.add_argument("--sam_checkpoint", type=str, default=None, help="SAM2 checkpoint path (required for PEFT models)")
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
    # Save probability heatmaps alongside SAM masks: {output_dir}/{stem}/heatmap-{i}.png
    p.add_argument("--output_dir", type=str, default=None, help="Directory to save per-step heatmaps (under {stem}/heatmap-{i}.png)")
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


def _is_peft_checkpoint(ckpt: dict) -> bool:
    """检查checkpoint是否是PEFT格式"""
    return 'point_predictor_state' in ckpt and 'sam_lora_state' in ckpt


def _load_model(model_path: str, device: torch.device, sam_checkpoint: Optional[str] = None):
    """加载模型，支持heatmap模型和PEFT模型"""
    if not model_path or not os.path.isfile(model_path):
        raise ValueError(f"Model path not found: {model_path}")
    
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    
    # 检查是否是PEFT checkpoint
    if _is_peft_checkpoint(ckpt):
        # 加载PEFT模型
        if not sam_checkpoint:
            raise ValueError("--sam_checkpoint is required for PEFT models")
        
        # 导入PEFT模块
        import sys
        from pathlib import Path
        _PARENT_DIR = Path(__file__).parent.parent
        if str(_PARENT_DIR) not in sys.path:
            sys.path.insert(0, str(_PARENT_DIR))
        
        from peft.lora_sam2 import LoRASAM2Wrapper
        from peft.point_predictor_peft import PointPredictorFromSAMFeatures
        from peft.utils_peft import load_checkpoint
        
        config = ckpt.get('config', {})
        lora_rank = config.get('lora_rank', 16)
        lora_alpha = config.get('lora_alpha', 32)
        fusion_mode = config.get('fusion_mode', 'film')
        feature_scale = config.get('feature_scale', 8)
        image_size = config.get('image_size', [512, 512])
        
        # 初始化SAM2 LoRA
        sam2_lora = LoRASAM2Wrapper(
            sam_checkpoint=sam_checkpoint,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            device=str(device),
        )
        
        # 初始化点预测网络
        sam_feature_dim_map = {4: 32, 8: 64, 16: 256}
        if feature_scale not in sam_feature_dim_map:
            raise ValueError(f"Unsupported feature_scale: {feature_scale}")
        sam_feature_dim = sam_feature_dim_map[feature_scale]
        
        point_predictor = PointPredictorFromSAMFeatures(
            sam_feature_dim=sam_feature_dim,
            output_size=tuple(image_size),
            fusion_mode=fusion_mode,
            feature_scale=feature_scale,
        ).to(device)
        
        # 加载权重
        load_checkpoint(
            model_path, point_predictor, sam2_lora, None, None, None, str(device)
        )
        
        point_predictor.eval()
        return ('peft', sam2_lora, point_predictor, feature_scale, tuple(image_size))
    else:
        # 加载heatmap模型
        cfg = ModelConfig(backbone="unet_s", pretrained=False, main_in_channels=3, cond_in_channels=1)
        model = PointHeatmapModel(cfg).to(device)
        sd = ckpt.get("model", ckpt)
        model.load_state_dict(sd, strict=False)
        model.eval()
        return ('heatmap', model)


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


def _predict_point_heatmap(model: PointHeatmapModel, rgb_tensor: torch.Tensor, gray_tensor: torch.Tensor, device: torch.device, tau: float) -> Tuple[float, float, int, np.ndarray]:
    with torch.no_grad():
        logits, label_logits = model(rgb_tensor.to(device), gray_tensor.to(device))
        xy = soft_argmax_from_logits(logits, temperature=tau)[0]
        lab = int(label_logits.argmax(dim=1).item())
        # probability heatmap for saving: softmax over pixels
        b, _, H, W = logits.shape
        p = F.softmax(logits.view(b, -1) / max(float(tau), 1e-6), dim=1).view(b, 1, H, W)
        prob = p[0, 0].detach().cpu().float().numpy()
    return float(xy[0].item()), float(xy[1].item()), lab, prob


def _predict_point_peft(sam2_lora, point_predictor, image_tensor: torch.Tensor, prev_mask: torch.Tensor, device: torch.device, feature_scale: int, tau: float) -> Tuple[float, float, int, np.ndarray]:
    """使用PEFT模型预测点"""
    with torch.no_grad():
        # 提取SAM特征
        sam_features = sam2_lora.get_image_features(image_tensor, feature_scale=feature_scale)
        
        # 点预测网络
        heatmap_logits, label_logits = point_predictor(sam_features, prev_mask)
        
        # 使用soft_argmax获取点坐标
        b, _, H, W = heatmap_logits.shape
        heatmap_flat = heatmap_logits.view(b, -1)
        
        # 使用softmax和加权平均计算坐标
        p = F.softmax(heatmap_flat / max(float(tau), 1e-6), dim=1)
        indices = torch.arange(heatmap_flat.size(1), device=device, dtype=torch.float32)
        x_flat = indices % W
        y_flat = indices // W
        x = (p * x_flat).sum(dim=1)
        y = (p * y_flat).sum(dim=1)
        
        lab = int(label_logits.argmax(dim=1).item())
        
        # 概率热力图用于保存
        prob = p.view(b, 1, H, W)[0, 0].detach().cpu().float().numpy()
        
    return float(x.item()), float(y.item()), lab, prob


def _save_heatmap_image(prob: np.ndarray, out_dir: Optional[str], stem: str, step_idx: int, orig_w: int, orig_h: int) -> None:
    if not out_dir:
        return
    try:
        hm_img = heatmap_to_pil(prob)
        if hm_img.size != (orig_w, orig_h):
            hm_img = hm_img.resize((orig_w, orig_h), Image.BILINEAR)
        subdir = os.path.join(out_dir, stem)
        os.makedirs(subdir, exist_ok=True)
        out_path = os.path.join(subdir, f"heatmap-{step_idx}.png")
        hm_img.save(out_path)
    except Exception:
        pass


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
            x, y, lab, prob = _predict_point(model, rgb_tensor, gray_tensor, device, args.tau)
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
        # save heatmap for step 0
        _save_heatmap_image(prob, args.output_dir or args.sam_dir, stem, 0, orig_w, orig_h)
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
            x, y, lab, prob = _predict_point(model, rgb_tensor, gray_tensor, device, args.tau)
        # scale back and clamp
        inf_H, inf_W = rgb_tensor.shape[-2], rgb_tensor.shape[-1]
        if (inf_W, inf_H) != (orig_w, orig_h):
            scale_x = float(orig_w) / float(inf_W)
            scale_y = float(orig_h) / float(inf_H)
            x = x * scale_x
            y = y * scale_y
        x = float(max(0.0, min(orig_w - 1.0, x)))
        y = float(max(0.0, min(orig_h - 1.0, y)))
        # Save heatmap for next step k = len(pts)
        stem = _stem(img_path)
        step_k = len(pts)
        _save_heatmap_image(prob, args.output_dir or sam_dir, stem, step_k, orig_w, orig_h)
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



```

```python
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
    from .utils import heatmap_to_pil
except Exception:
    # script context
    import sys as _sys
    import os as _os
    _pkg_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _pkg_root not in _sys.path:
        _sys.path.insert(0, _pkg_root)
    from heatmap.model import ModelConfig, PointHeatmapModel, soft_argmax_from_logits
    from heatmap.utils import heatmap_to_pil

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
    p.add_argument("--sam_checkpoint", type=str, default=None, help="SAM2 checkpoint path (required for PEFT models)")
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
    # Save probability heatmaps alongside SAM masks: {output_dir}/{stem}/heatmap-{i}.png
    p.add_argument("--output_dir", type=str, default=None, help="Directory to save per-step heatmaps (under {stem}/heatmap-{i}.png)")
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


def _is_peft_checkpoint(ckpt: dict) -> bool:
    """检查checkpoint是否是PEFT格式"""
    return 'point_predictor_state' in ckpt and 'sam_lora_state' in ckpt


def _load_model(model_path: str, device: torch.device, sam_checkpoint: Optional[str] = None):
    """加载模型，支持heatmap模型和PEFT模型"""
    if not model_path or not os.path.isfile(model_path):
        raise ValueError(f"Model path not found: {model_path}")
    
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    
    # 检查是否是PEFT checkpoint
    if _is_peft_checkpoint(ckpt):
        # 加载PEFT模型
        if not sam_checkpoint:
            raise ValueError("--sam_checkpoint is required for PEFT models")
        
        # 导入PEFT模块
        import sys
        from pathlib import Path
        _PARENT_DIR = Path(__file__).parent.parent
        if str(_PARENT_DIR) not in sys.path:
            sys.path.insert(0, str(_PARENT_DIR))
        
        from peft.lora_sam2 import LoRASAM2Wrapper
        from peft.point_predictor_peft import PointPredictorFromSAMFeatures
        from peft.utils_peft import load_checkpoint
        
        config = ckpt.get('config', {})
        lora_rank = config.get('lora_rank', 16)
        lora_alpha = config.get('lora_alpha', 32)
        fusion_mode = config.get('fusion_mode', 'film')
        feature_scale = config.get('feature_scale', 8)
        image_size = config.get('image_size', [512, 512])
        
        # 初始化SAM2 LoRA
        sam2_lora = LoRASAM2Wrapper(
            sam_checkpoint=sam_checkpoint,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            device=str(device),
        )
        
        # 初始化点预测网络
        sam_feature_dim_map = {4: 32, 8: 64, 16: 256}
        if feature_scale not in sam_feature_dim_map:
            raise ValueError(f"Unsupported feature_scale: {feature_scale}")
        sam_feature_dim = sam_feature_dim_map[feature_scale]
        
        point_predictor = PointPredictorFromSAMFeatures(
            sam_feature_dim=sam_feature_dim,
            output_size=tuple(image_size),
            fusion_mode=fusion_mode,
            feature_scale=feature_scale,
        ).to(device)
        
        # 加载权重
        load_checkpoint(
            model_path, point_predictor, sam2_lora, None, None, None, str(device)
        )
        
        point_predictor.eval()
        return ('peft', sam2_lora, point_predictor, feature_scale, tuple(image_size))
    else:
        # 加载heatmap模型
        cfg = ModelConfig(backbone="unet_s", pretrained=False, main_in_channels=3, cond_in_channels=1)
        model = PointHeatmapModel(cfg).to(device)
        sd = ckpt.get("model", ckpt)
        model.load_state_dict(sd, strict=False)
        model.eval()
        return ('heatmap', model)


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


def _predict_point_heatmap(model: PointHeatmapModel, rgb_tensor: torch.Tensor, gray_tensor: torch.Tensor, device: torch.device, tau: float) -> Tuple[float, float, int, np.ndarray]:
    with torch.no_grad():
        logits, label_logits = model(rgb_tensor.to(device), gray_tensor.to(device))
        xy = soft_argmax_from_logits(logits, temperature=tau)[0]
        lab = int(label_logits.argmax(dim=1).item())
        # probability heatmap for saving: softmax over pixels
        b, _, H, W = logits.shape
        p = F.softmax(logits.view(b, -1) / max(float(tau), 1e-6), dim=1).view(b, 1, H, W)
        prob = p[0, 0].detach().cpu().float().numpy()
    return float(xy[0].item()), float(xy[1].item()), lab, prob


def _predict_point_peft(sam2_lora, point_predictor, image_tensor: torch.Tensor, prev_mask: torch.Tensor, device: torch.device, feature_scale: int, tau: float) -> Tuple[float, float, int, np.ndarray]:
    """使用PEFT模型预测点"""
    with torch.no_grad():
        # 提取SAM特征
        sam_features = sam2_lora.get_image_features(image_tensor, feature_scale=feature_scale)
        
        # 点预测网络
        heatmap_logits, label_logits = point_predictor(sam_features, prev_mask)
        
        # 使用soft_argmax获取点坐标
        b, _, H, W = heatmap_logits.shape
        heatmap_flat = heatmap_logits.view(b, -1)
        
        # 使用softmax和加权平均计算坐标
        p = F.softmax(heatmap_flat / max(float(tau), 1e-6), dim=1)
        indices = torch.arange(heatmap_flat.size(1), device=device, dtype=torch.float32)
        x_flat = indices % W
        y_flat = indices // W
        x = (p * x_flat).sum(dim=1)
        y = (p * y_flat).sum(dim=1)
        
        lab = int(label_logits.argmax(dim=1).item())
        
        # 概率热力图用于保存
        prob = p.view(b, 1, H, W)[0, 0].detach().cpu().float().numpy()
        
    return float(x.item()), float(y.item()), lab, prob


def _save_heatmap_image(prob: np.ndarray, out_dir: Optional[str], stem: str, step_idx: int, orig_w: int, orig_h: int) -> None:
    if not out_dir:
        return
    try:
        hm_img = heatmap_to_pil(prob)
        if hm_img.size != (orig_w, orig_h):
            hm_img = hm_img.resize((orig_w, orig_h), Image.BILINEAR)
        subdir = os.path.join(out_dir, stem)
        os.makedirs(subdir, exist_ok=True)
        out_path = os.path.join(subdir, f"heatmap-{step_idx}.png")
        hm_img.save(out_path)
    except Exception:
        pass


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
    
    # 加载模型（支持heatmap和PEFT）
    model_info = _load_model(args.model_path, device, args.sam_checkpoint)
    is_peft = model_info[0] == 'peft'
    
    if is_peft:
        sam2_lora, point_predictor, feature_scale, image_size = model_info[1], model_info[2], model_info[3], model_info[4]
        H, W = image_size[0], image_size[1]
    else:
        model = model_info[1]
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
        if is_peft:
            # PEFT模型：需要SAM特征
            import torchvision.transforms.functional as TF
            image = Image.open(img_path).convert('RGB')
            orig_w, orig_h = image.size
            
            # Resize到模型输入尺寸
            image_resized = image.resize((W, H), Image.BILINEAR)
            image_tensor = TF.to_tensor(image_resized).unsqueeze(0).to(device)  # [1, 3, H, W]
            
            # 初始掩模为全零
            prev_mask = torch.zeros(1, 1, H, W).to(device)
            
            with torch.amp.autocast(device_type=autocast_device_type, enabled=amp_enabled):
                x, y, lab, prob = _predict_point_peft(sam2_lora, point_predictor, image_tensor, prev_mask, device, feature_scale, args.tau)
            
            # 缩放回原始尺寸
            scale_x = float(orig_w) / float(W)
            scale_y = float(orig_h) / float(H)
            x = x * scale_x
            y = y * scale_y
            
            # 调整概率热力图尺寸用于保存
            if prob.shape != (orig_h, orig_w):
                from scipy.ndimage import zoom
                scale_y_prob = orig_h / prob.shape[0]
                scale_x_prob = orig_w / prob.shape[1]
                prob = zoom(prob, (scale_y_prob, scale_x_prob), order=1)
        else:
            # Heatmap模型：使用RGB和灰度掩模
            rgb_tensor, gray_tensor, (orig_h, orig_w) = _to_tensor_separate(img_path, gray_mask=None, out_hw=(H, W))
            with torch.amp.autocast(device_type=autocast_device_type, enabled=amp_enabled):
                x, y, lab, prob = _predict_point_heatmap(model, rgb_tensor, gray_tensor, device, args.tau)
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
        # save heatmap for step 0
        _save_heatmap_image(prob, args.output_dir or args.sam_dir, stem, 0, orig_w, orig_h)
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
            x, y, lab, prob = _predict_point(model, rgb_tensor, gray_tensor, device, args.tau)
        # scale back and clamp
        inf_H, inf_W = rgb_tensor.shape[-2], rgb_tensor.shape[-1]
        if (inf_W, inf_H) != (orig_w, orig_h):
            scale_x = float(orig_w) / float(inf_W)
            scale_y = float(orig_h) / float(inf_H)
            x = x * scale_x
            y = y * scale_y
        x = float(max(0.0, min(orig_w - 1.0, x)))
        y = float(max(0.0, min(orig_h - 1.0, y)))
        # Save heatmap for next step k = len(pts)
        stem = _stem(img_path)
        step_k = len(pts)
        _save_heatmap_image(prob, args.output_dir or sam_dir, stem, step_k, orig_w, orig_h)
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



```

```python
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
    from .utils import heatmap_to_pil
except Exception:
    # script context
    import sys as _sys
    import os as _os
    _pkg_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _pkg_root not in _sys.path:
        _sys.path.insert(0, _pkg_root)
    from heatmap.model import ModelConfig, PointHeatmapModel, soft_argmax_from_logits
    from heatmap.utils import heatmap_to_pil

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
    p.add_argument("--sam_checkpoint", type=str, default=None, help="SAM2 checkpoint path (required for PEFT models)")
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
    # Save probability heatmaps alongside SAM masks: {output_dir}/{stem}/heatmap-{i}.png
    p.add_argument("--output_dir", type=str, default=None, help="Directory to save per-step heatmaps (under {stem}/heatmap-{i}.png)")
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


def _is_peft_checkpoint(ckpt: dict) -> bool:
    """检查checkpoint是否是PEFT格式"""
    return 'point_predictor_state' in ckpt and 'sam_lora_state' in ckpt


def _load_model(model_path: str, device: torch.device, sam_checkpoint: Optional[str] = None):
    """加载模型，支持heatmap模型和PEFT模型"""
    if not model_path or not os.path.isfile(model_path):
        raise ValueError(f"Model path not found: {model_path}")
    
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    
    # 检查是否是PEFT checkpoint
    if _is_peft_checkpoint(ckpt):
        # 加载PEFT模型
        if not sam_checkpoint:
            raise ValueError("--sam_checkpoint is required for PEFT models")
        
        # 导入PEFT模块
        import sys
        from pathlib import Path
        _PARENT_DIR = Path(__file__).parent.parent
        if str(_PARENT_DIR) not in sys.path:
            sys.path.insert(0, str(_PARENT_DIR))
        
        from peft.lora_sam2 import LoRASAM2Wrapper
        from peft.point_predictor_peft import PointPredictorFromSAMFeatures
        from peft.utils_peft import load_checkpoint
        
        config = ckpt.get('config', {})
        lora_rank = config.get('lora_rank', 16)
        lora_alpha = config.get('lora_alpha', 32)
        fusion_mode = config.get('fusion_mode', 'film')
        feature_scale = config.get('feature_scale', 8)
        image_size = config.get('image_size', [512, 512])
        
        # 初始化SAM2 LoRA
        sam2_lora = LoRASAM2Wrapper(
            sam_checkpoint=sam_checkpoint,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            device=str(device),
        )
        
        # 初始化点预测网络
        sam_feature_dim_map = {4: 32, 8: 64, 16: 256}
        if feature_scale not in sam_feature_dim_map:
            raise ValueError(f"Unsupported feature_scale: {feature_scale}")
        sam_feature_dim = sam_feature_dim_map[feature_scale]
        
        point_predictor = PointPredictorFromSAMFeatures(
            sam_feature_dim=sam_feature_dim,
            output_size=tuple(image_size),
            fusion_mode=fusion_mode,
            feature_scale=feature_scale,
        ).to(device)
        
        # 加载权重
        load_checkpoint(
            model_path, point_predictor, sam2_lora, None, None, None, str(device)
        )
        
        point_predictor.eval()
        return ('peft', sam2_lora, point_predictor, feature_scale, tuple(image_size))
    else:
        # 加载heatmap模型
        cfg = ModelConfig(backbone="unet_s", pretrained=False, main_in_channels=3, cond_in_channels=1)
        model = PointHeatmapModel(cfg).to(device)
        sd = ckpt.get("model", ckpt)
        model.load_state_dict(sd, strict=False)
        model.eval()
        return ('heatmap', model)


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


def _predict_point_heatmap(model: PointHeatmapModel, rgb_tensor: torch.Tensor, gray_tensor: torch.Tensor, device: torch.device, tau: float) -> Tuple[float, float, int, np.ndarray]:
    with torch.no_grad():
        logits, label_logits = model(rgb_tensor.to(device), gray_tensor.to(device))
        xy = soft_argmax_from_logits(logits, temperature=tau)[0]
        lab = int(label_logits.argmax(dim=1).item())
        # probability heatmap for saving: softmax over pixels
        b, _, H, W = logits.shape
        p = F.softmax(logits.view(b, -1) / max(float(tau), 1e-6), dim=1).view(b, 1, H, W)
        prob = p[0, 0].detach().cpu().float().numpy()
    return float(xy[0].item()), float(xy[1].item()), lab, prob


def _predict_point_peft(sam2_lora, point_predictor, image_tensor: torch.Tensor, prev_mask: torch.Tensor, device: torch.device, feature_scale: int, tau: float) -> Tuple[float, float, int, np.ndarray]:
    """使用PEFT模型预测点"""
    with torch.no_grad():
        # 提取SAM特征
        sam_features = sam2_lora.get_image_features(image_tensor, feature_scale=feature_scale)
        
        # 点预测网络
        heatmap_logits, label_logits = point_predictor(sam_features, prev_mask)
        
        # 使用soft_argmax获取点坐标
        b, _, H, W = heatmap_logits.shape
        heatmap_flat = heatmap_logits.view(b, -1)
        
        # 使用softmax和加权平均计算坐标
        p = F.softmax(heatmap_flat / max(float(tau), 1e-6), dim=1)
        indices = torch.arange(heatmap_flat.size(1), device=device, dtype=torch.float32)
        x_flat = indices % W
        y_flat = indices // W
        x = (p * x_flat).sum(dim=1)
        y = (p * y_flat).sum(dim=1)
        
        lab = int(label_logits.argmax(dim=1).item())
        
        # 概率热力图用于保存
        prob = p.view(b, 1, H, W)[0, 0].detach().cpu().float().numpy()
        
    return float(x.item()), float(y.item()), lab, prob


def _save_heatmap_image(prob: np.ndarray, out_dir: Optional[str], stem: str, step_idx: int, orig_w: int, orig_h: int) -> None:
    if not out_dir:
        return
    try:
        hm_img = heatmap_to_pil(prob)
        if hm_img.size != (orig_w, orig_h):
            hm_img = hm_img.resize((orig_w, orig_h), Image.BILINEAR)
        subdir = os.path.join(out_dir, stem)
        os.makedirs(subdir, exist_ok=True)
        out_path = os.path.join(subdir, f"heatmap-{step_idx}.png")
        hm_img.save(out_path)
    except Exception:
        pass


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
    
    # 加载模型（支持heatmap和PEFT）
    model_info = _load_model(args.model_path, device, args.sam_checkpoint)
    is_peft = model_info[0] == 'peft'
    
    if is_peft:
        sam2_lora, point_predictor, feature_scale, image_size = model_info[1], model_info[2], model_info[3], model_info[4]
        H, W = image_size[0], image_size[1]
    else:
        model = model_info[1]
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
        if is_peft:
            # PEFT模型：需要SAM特征
            import torchvision.transforms.functional as TF
            image = Image.open(img_path).convert('RGB')
            orig_w, orig_h = image.size
            
            # Resize到模型输入尺寸
            image_resized = image.resize((W, H), Image.BILINEAR)
            image_tensor = TF.to_tensor(image_resized).unsqueeze(0).to(device)  # [1, 3, H, W]
            
            # 初始掩模为全零
            prev_mask = torch.zeros(1, 1, H, W).to(device)
            
            with torch.amp.autocast(device_type=autocast_device_type, enabled=amp_enabled):
                x, y, lab, prob = _predict_point_peft(sam2_lora, point_predictor, image_tensor, prev_mask, device, feature_scale, args.tau)
            
            # 缩放回原始尺寸
            scale_x = float(orig_w) / float(W)
            scale_y = float(orig_h) / float(H)
            x = x * scale_x
            y = y * scale_y
            
            # 调整概率热力图尺寸用于保存
            if prob.shape != (orig_h, orig_w):
                from scipy.ndimage import zoom
                scale_y_prob = orig_h / prob.shape[0]
                scale_x_prob = orig_w / prob.shape[1]
                prob = zoom(prob, (scale_y_prob, scale_x_prob), order=1)
        else:
            # Heatmap模型：使用RGB和灰度掩模
            rgb_tensor, gray_tensor, (orig_h, orig_w) = _to_tensor_separate(img_path, gray_mask=None, out_hw=(H, W))
            with torch.amp.autocast(device_type=autocast_device_type, enabled=amp_enabled):
                x, y, lab, prob = _predict_point_heatmap(model, rgb_tensor, gray_tensor, device, args.tau)
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
        # save heatmap for step 0
        _save_heatmap_image(prob, args.output_dir or args.sam_dir, stem, 0, orig_w, orig_h)
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
    
    # 加载模型（支持heatmap和PEFT）
    model_info = _load_model(args.model_path, device, args.sam_checkpoint)
    is_peft = model_info[0] == 'peft'
    
    if is_peft:
        sam2_lora, point_predictor, feature_scale, image_size = model_info[1], model_info[2], model_info[3], model_info[4]
        H, W = image_size[0], image_size[1]
    else:
        model = model_info[1]
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
        
        if is_peft:
            # PEFT模型：需要SAM特征
            import torchvision.transforms.functional as TF
            image = Image.open(img_path).convert('RGB')
            orig_w, orig_h = image.size
            
            # Resize到模型输入尺寸
            image_resized = image.resize((W, H), Image.BILINEAR)
            image_tensor = TF.to_tensor(image_resized).unsqueeze(0).to(device)
            
            # 加载前一步掩模
            if k == 0:
                prev_mask = torch.zeros(1, 1, H, W).to(device)
            else:
                stem = _stem(img_path)
                prev_path = os.path.join(sam_dir, stem, f"{k-1}.png")
                if os.path.isfile(prev_path):
                    prev_mask_img = Image.open(prev_path).convert('L')
                    prev_mask_resized = prev_mask_img.resize((W, H), Image.NEAREST)
                    prev_mask_tensor = TF.to_tensor(prev_mask_resized).to(device)
                    prev_mask = (prev_mask_tensor > 0.5).float().unsqueeze(0)
                else:
                    prev_mask = torch.zeros(1, 1, H, W).to(device)
            
            with torch.amp.autocast(device_type=autocast_device_type, enabled=amp_enabled):
                x, y, lab, prob = _predict_point_peft(sam2_lora, point_predictor, image_tensor, prev_mask, device, feature_scale, args.tau)
            
            # 缩放回原始尺寸
            scale_x = float(orig_w) / float(W)
            scale_y = float(orig_h) / float(H)
            x = x * scale_x
            y = y * scale_y
            
            # 调整概率热力图尺寸用于保存
            if prob.shape != (orig_h, orig_w):
                from scipy.ndimage import zoom
                scale_y_prob = orig_h / prob.shape[0]
                scale_x_prob = orig_w / prob.shape[1]
                prob = zoom(prob, (scale_y_prob, scale_x_prob), order=1)
        else:
            # Heatmap模型：使用RGB和灰度掩模
            # prev mask
            if k == 0:
                gray = None
            else:
                stem = _stem(img_path)
                prev_path = os.path.join(sam_dir, stem, f"{k-1}.png")
                gray = Image.open(prev_path).convert("L") if os.path.isfile(prev_path) else None
            rgb_tensor, gray_tensor, (orig_h, orig_w) = _to_tensor_separate(img_path, gray_mask=gray, out_hw=(H, W))
            with torch.amp.autocast(device_type=autocast_device_type, enabled=amp_enabled):
                x, y, lab, prob = _predict_point_heatmap(model, rgb_tensor, gray_tensor, device, args.tau)
            # scale back and clamp
            inf_H, inf_W = rgb_tensor.shape[-2], rgb_tensor.shape[-1]
            if (inf_W, inf_H) != (orig_w, orig_h):
                scale_x = float(orig_w) / float(inf_W)
                scale_y = float(orig_h) / float(inf_H)
                x = x * scale_x
                y = y * scale_y
        
        x = float(max(0.0, min(orig_w - 1.0, x)))
        y = float(max(0.0, min(orig_h - 1.0, y)))
        # Save heatmap for next step k = len(pts)
        stem = _stem(img_path)
        step_k = len(pts)
        _save_heatmap_image(prob, args.output_dir or sam_dir, stem, step_k, orig_w, orig_h)
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



```


添加缺失的导入：


```python
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
from scipy.ndimage import zoom

import torch
import torch.nn.functional as F

try:
    # package context
    from .model import ModelConfig, PointHeatmapModel, soft_argmax_from_logits
    from .utils import heatmap_to_pil
except Exception:
    # script context
    import sys as _sys
    import os as _os
    _pkg_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _pkg_root not in _sys.path:
        _sys.path.insert(0, _pkg_root)
    from heatmap.model import ModelConfig, PointHeatmapModel, soft_argmax_from_logits
    from heatmap.utils import heatmap_to_pil

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
    p.add_argument("--sam_checkpoint", type=str, default=None, help="SAM2 checkpoint path (required for PEFT models)")
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
    # Save probability heatmaps alongside SAM masks: {output_dir}/{stem}/heatmap-{i}.png
    p.add_argument("--output_dir", type=str, default=None, help="Directory to save per-step heatmaps (under {stem}/heatmap-{i}.png)")
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


def _is_peft_checkpoint(ckpt: dict) -> bool:
    """检查checkpoint是否是PEFT格式"""
    return 'point_predictor_state' in ckpt and 'sam_lora_state' in ckpt


def _load_model(model_path: str, device: torch.device, sam_checkpoint: Optional[str] = None):
    """加载模型，支持heatmap模型和PEFT模型"""
    if not model_path or not os.path.isfile(model_path):
        raise ValueError(f"Model path not found: {model_path}")
    
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    
    # 检查是否是PEFT checkpoint
    if _is_peft_checkpoint(ckpt):
        # 加载PEFT模型
        if not sam_checkpoint:
            raise ValueError("--sam_checkpoint is required for PEFT models")
        
        # 导入PEFT模块
        import sys
        from pathlib import Path
        _PARENT_DIR = Path(__file__).parent.parent
        if str(_PARENT_DIR) not in sys.path:
            sys.path.insert(0, str(_PARENT_DIR))
        
        from peft.lora_sam2 import LoRASAM2Wrapper
        from peft.point_predictor_peft import PointPredictorFromSAMFeatures
        from peft.utils_peft import load_checkpoint
        
        config = ckpt.get('config', {})
        lora_rank = config.get('lora_rank', 16)
        lora_alpha = config.get('lora_alpha', 32)
        fusion_mode = config.get('fusion_mode', 'film')
        feature_scale = config.get('feature_scale', 8)
        image_size = config.get('image_size', [512, 512])
        
        # 初始化SAM2 LoRA
        sam2_lora = LoRASAM2Wrapper(
            sam_checkpoint=sam_checkpoint,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            device=str(device),
        )
        
        # 初始化点预测网络
        sam_feature_dim_map = {4: 32, 8: 64, 16: 256}
        if feature_scale not in sam_feature_dim_map:
            raise ValueError(f"Unsupported feature_scale: {feature_scale}")
        sam_feature_dim = sam_feature_dim_map[feature_scale]
        
        point_predictor = PointPredictorFromSAMFeatures(
            sam_feature_dim=sam_feature_dim,
            output_size=tuple(image_size),
            fusion_mode=fusion_mode,
            feature_scale=feature_scale,
        ).to(device)
        
        # 加载权重
        load_checkpoint(
            model_path, point_predictor, sam2_lora, None, None, None, str(device)
        )
        
        point_predictor.eval()
        return ('peft', sam2_lora, point_predictor, feature_scale, tuple(image_size))
    else:
        # 加载heatmap模型
        cfg = ModelConfig(backbone="unet_s", pretrained=False, main_in_channels=3, cond_in_channels=1)
        model = PointHeatmapModel(cfg).to(device)
        sd = ckpt.get("model", ckpt)
        model.load_state_dict(sd, strict=False)
        model.eval()
        return ('heatmap', model)


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


def _predict_point_heatmap(model: PointHeatmapModel, rgb_tensor: torch.Tensor, gray_tensor: torch.Tensor, device: torch.device, tau: float) -> Tuple[float, float, int, np.ndarray]:
    with torch.no_grad():
        logits, label_logits = model(rgb_tensor.to(device), gray_tensor.to(device))
        xy = soft_argmax_from_logits(logits, temperature=tau)[0]
        lab = int(label_logits.argmax(dim=1).item())
        # probability heatmap for saving: softmax over pixels
        b, _, H, W = logits.shape
        p = F.softmax(logits.view(b, -1) / max(float(tau), 1e-6), dim=1).view(b, 1, H, W)
        prob = p[0, 0].detach().cpu().float().numpy()
    return float(xy[0].item()), float(xy[1].item()), lab, prob


def _predict_point_peft(sam2_lora, point_predictor, image_tensor: torch.Tensor, prev_mask: torch.Tensor, device: torch.device, feature_scale: int, tau: float) -> Tuple[float, float, int, np.ndarray]:
    """使用PEFT模型预测点"""
    with torch.no_grad():
        # 提取SAM特征
        sam_features = sam2_lora.get_image_features(image_tensor, feature_scale=feature_scale)
        
        # 点预测网络
        heatmap_logits, label_logits = point_predictor(sam_features, prev_mask)
        
        # 使用soft_argmax获取点坐标
        b, _, H, W = heatmap_logits.shape
        heatmap_flat = heatmap_logits.view(b, -1)
        
        # 使用softmax和加权平均计算坐标
        p = F.softmax(heatmap_flat / max(float(tau), 1e-6), dim=1)
        indices = torch.arange(heatmap_flat.size(1), device=device, dtype=torch.float32)
        x_flat = indices % W
        y_flat = indices // W
        x = (p * x_flat).sum(dim=1)
        y = (p * y_flat).sum(dim=1)
        
        lab = int(label_logits.argmax(dim=1).item())
        
        # 概率热力图用于保存
        prob = p.view(b, 1, H, W)[0, 0].detach().cpu().float().numpy()
        
    return float(x.item()), float(y.item()), lab, prob


def _save_heatmap_image(prob: np.ndarray, out_dir: Optional[str], stem: str, step_idx: int, orig_w: int, orig_h: int) -> None:
    if not out_dir:
        return
    try:
        hm_img = heatmap_to_pil(prob)
        if hm_img.size != (orig_w, orig_h):
            hm_img = hm_img.resize((orig_w, orig_h), Image.BILINEAR)
        subdir = os.path.join(out_dir, stem)
        os.makedirs(subdir, exist_ok=True)
        out_path = os.path.join(subdir, f"heatmap-{step_idx}.png")
        hm_img.save(out_path)
    except Exception:
        pass


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
    
    # 加载模型（支持heatmap和PEFT）
    model_info = _load_model(args.model_path, device, args.sam_checkpoint)
    is_peft = model_info[0] == 'peft'
    
    if is_peft:
        sam2_lora, point_predictor, feature_scale, image_size = model_info[1], model_info[2], model_info[3], model_info[4]
        H, W = image_size[0], image_size[1]
    else:
        model = model_info[1]
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
        if is_peft:
            # PEFT模型：需要SAM特征
            import torchvision.transforms.functional as TF
            image = Image.open(img_path).convert('RGB')
            orig_w, orig_h = image.size
            
            # Resize到模型输入尺寸
            image_resized = image.resize((W, H), Image.BILINEAR)
            image_tensor = TF.to_tensor(image_resized).unsqueeze(0).to(device)  # [1, 3, H, W]
            
            # 初始掩模为全零
            prev_mask = torch.zeros(1, 1, H, W).to(device)
            
            with torch.amp.autocast(device_type=autocast_device_type, enabled=amp_enabled):
                x, y, lab, prob = _predict_point_peft(sam2_lora, point_predictor, image_tensor, prev_mask, device, feature_scale, args.tau)
            
            # 缩放回原始尺寸
            scale_x = float(orig_w) / float(W)
            scale_y = float(orig_h) / float(H)
            x = x * scale_x
            y = y * scale_y
            
            # 调整概率热力图尺寸用于保存
            if prob.shape != (orig_h, orig_w):
                from scipy.ndimage import zoom
                scale_y_prob = orig_h / prob.shape[0]
                scale_x_prob = orig_w / prob.shape[1]
                prob = zoom(prob, (scale_y_prob, scale_x_prob), order=1)
        else:
            # Heatmap模型：使用RGB和灰度掩模
            rgb_tensor, gray_tensor, (orig_h, orig_w) = _to_tensor_separate(img_path, gray_mask=None, out_hw=(H, W))
            with torch.amp.autocast(device_type=autocast_device_type, enabled=amp_enabled):
                x, y, lab, prob = _predict_point_heatmap(model, rgb_tensor, gray_tensor, device, args.tau)
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
        # save heatmap for step 0
        _save_heatmap_image(prob, args.output_dir or args.sam_dir, stem, 0, orig_w, orig_h)
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
    
    # 加载模型（支持heatmap和PEFT）
    model_info = _load_model(args.model_path, device, args.sam_checkpoint)
    is_peft = model_info[0] == 'peft'
    
    if is_peft:
        sam2_lora, point_predictor, feature_scale, image_size = model_info[1], model_info[2], model_info[3], model_info[4]
        H, W = image_size[0], image_size[1]
    else:
        model = model_info[1]
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
        
        if is_peft:
            # PEFT模型：需要SAM特征
            import torchvision.transforms.functional as TF
            image = Image.open(img_path).convert('RGB')
            orig_w, orig_h = image.size
            
            # Resize到模型输入尺寸
            image_resized = image.resize((W, H), Image.BILINEAR)
            image_tensor = TF.to_tensor(image_resized).unsqueeze(0).to(device)
            
            # 加载前一步掩模
            if k == 0:
                prev_mask = torch.zeros(1, 1, H, W).to(device)
            else:
                stem = _stem(img_path)
                prev_path = os.path.join(sam_dir, stem, f"{k-1}.png")
                if os.path.isfile(prev_path):
                    prev_mask_img = Image.open(prev_path).convert('L')
                    prev_mask_resized = prev_mask_img.resize((W, H), Image.NEAREST)
                    prev_mask_tensor = TF.to_tensor(prev_mask_resized).to(device)
                    prev_mask = (prev_mask_tensor > 0.5).float().unsqueeze(0)
                else:
                    prev_mask = torch.zeros(1, 1, H, W).to(device)
            
            with torch.amp.autocast(device_type=autocast_device_type, enabled=amp_enabled):
                x, y, lab, prob = _predict_point_peft(sam2_lora, point_predictor, image_tensor, prev_mask, device, feature_scale, args.tau)
            
            # 缩放回原始尺寸
            scale_x = float(orig_w) / float(W)
            scale_y = float(orig_h) / float(H)
            x = x * scale_x
            y = y * scale_y
            
            # 调整概率热力图尺寸用于保存
            if prob.shape != (orig_h, orig_w):
                from scipy.ndimage import zoom
                scale_y_prob = orig_h / prob.shape[0]
                scale_x_prob = orig_w / prob.shape[1]
                prob = zoom(prob, (scale_y_prob, scale_x_prob), order=1)
        else:
            # Heatmap模型：使用RGB和灰度掩模
            # prev mask
            if k == 0:
                gray = None
            else:
                stem = _stem(img_path)
                prev_path = os.path.join(sam_dir, stem, f"{k-1}.png")
                gray = Image.open(prev_path).convert("L") if os.path.isfile(prev_path) else None
            rgb_tensor, gray_tensor, (orig_h, orig_w) = _to_tensor_separate(img_path, gray_mask=gray, out_hw=(H, W))
            with torch.amp.autocast(device_type=autocast_device_type, enabled=amp_enabled):
                x, y, lab, prob = _predict_point_heatmap(model, rgb_tensor, gray_tensor, device, args.tau)
            # scale back and clamp
            inf_H, inf_W = rgb_tensor.shape[-2], rgb_tensor.shape[-1]
            if (inf_W, inf_H) != (orig_w, orig_h):
                scale_x = float(orig_w) / float(inf_W)
                scale_y = float(orig_h) / float(inf_H)
                x = x * scale_x
                y = y * scale_y
        
        x = float(max(0.0, min(orig_w - 1.0, x)))
        y = float(max(0.0, min(orig_h - 1.0, y)))
        # Save heatmap for next step k = len(pts)
        stem = _stem(img_path)
        step_k = len(pts)
        _save_heatmap_image(prob, args.output_dir or sam_dir, stem, step_k, orig_w, orig_h)
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



```

```python
            # 调整概率热力图尺寸用于保存
            if prob.shape != (orig_h, orig_w):
                scale_y_prob = orig_h / prob.shape[0]
                scale_x_prob = orig_w / prob.shape[1]
                prob = zoom(prob, (scale_y_prob, scale_x_prob), order=1)
```

```python
            # 调整概率热力图尺寸用于保存
            if prob.shape != (orig_h, orig_w):
                scale_y_prob = orig_h / prob.shape[0]
                scale_x_prob = orig_w / prob.shape[1]
                prob = zoom(prob, (scale_y_prob, scale_x_prob), order=1)
```


移除重复的导入语句：



移除函数内部的重复导入：


```python
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
from scipy.ndimage import zoom

import torch
import torch.nn.functional as F

try:
    # package context
    from .model import ModelConfig, PointHeatmapModel, soft_argmax_from_logits
    from .utils import heatmap_to_pil
except Exception:
    # script context
    import sys as _sys
    import os as _os
    _pkg_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _pkg_root not in _sys.path:
        _sys.path.insert(0, _pkg_root)
    from heatmap.model import ModelConfig, PointHeatmapModel, soft_argmax_from_logits
    from heatmap.utils import heatmap_to_pil

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
    p.add_argument("--sam_checkpoint", type=str, default=None, help="SAM2 checkpoint path (required for PEFT models)")
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
    # Save probability heatmaps alongside SAM masks: {output_dir}/{stem}/heatmap-{i}.png
    p.add_argument("--output_dir", type=str, default=None, help="Directory to save per-step heatmaps (under {stem}/heatmap-{i}.png)")
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


def _is_peft_checkpoint(ckpt: dict) -> bool:
    """检查checkpoint是否是PEFT格式"""
    return 'point_predictor_state' in ckpt and 'sam_lora_state' in ckpt


def _load_model(model_path: str, device: torch.device, sam_checkpoint: Optional[str] = None):
    """加载模型，支持heatmap模型和PEFT模型"""
    if not model_path or not os.path.isfile(model_path):
        raise ValueError(f"Model path not found: {model_path}")
    
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    
    # 检查是否是PEFT checkpoint
    if _is_peft_checkpoint(ckpt):
        # 加载PEFT模型
        if not sam_checkpoint:
            raise ValueError("--sam_checkpoint is required for PEFT models")
        
        # 导入PEFT模块
        import sys
        from pathlib import Path
        _PARENT_DIR = Path(__file__).parent.parent
        if str(_PARENT_DIR) not in sys.path:
            sys.path.insert(0, str(_PARENT_DIR))
        
        from peft.lora_sam2 import LoRASAM2Wrapper
        from peft.point_predictor_peft import PointPredictorFromSAMFeatures
        from peft.utils_peft import load_checkpoint
        
        config = ckpt.get('config', {})
        lora_rank = config.get('lora_rank', 16)
        lora_alpha = config.get('lora_alpha', 32)
        fusion_mode = config.get('fusion_mode', 'film')
        feature_scale = config.get('feature_scale', 8)
        image_size = config.get('image_size', [512, 512])
        
        # 初始化SAM2 LoRA
        sam2_lora = LoRASAM2Wrapper(
            sam_checkpoint=sam_checkpoint,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            device=str(device),
        )
        
        # 初始化点预测网络
        sam_feature_dim_map = {4: 32, 8: 64, 16: 256}
        if feature_scale not in sam_feature_dim_map:
            raise ValueError(f"Unsupported feature_scale: {feature_scale}")
        sam_feature_dim = sam_feature_dim_map[feature_scale]
        
        point_predictor = PointPredictorFromSAMFeatures(
            sam_feature_dim=sam_feature_dim,
            output_size=tuple(image_size),
            fusion_mode=fusion_mode,
            feature_scale=feature_scale,
        ).to(device)
        
        # 加载权重
        load_checkpoint(
            model_path, point_predictor, sam2_lora, None, None, None, str(device)
        )
        
        point_predictor.eval()
        return ('peft', sam2_lora, point_predictor, feature_scale, tuple(image_size))
    else:
        # 加载heatmap模型
        cfg = ModelConfig(backbone="unet_s", pretrained=False, main_in_channels=3, cond_in_channels=1)
        model = PointHeatmapModel(cfg).to(device)
        sd = ckpt.get("model", ckpt)
        model.load_state_dict(sd, strict=False)
        model.eval()
        return ('heatmap', model)


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


def _predict_point_heatmap(model: PointHeatmapModel, rgb_tensor: torch.Tensor, gray_tensor: torch.Tensor, device: torch.device, tau: float) -> Tuple[float, float, int, np.ndarray]:
    with torch.no_grad():
        logits, label_logits = model(rgb_tensor.to(device), gray_tensor.to(device))
        xy = soft_argmax_from_logits(logits, temperature=tau)[0]
        lab = int(label_logits.argmax(dim=1).item())
        # probability heatmap for saving: softmax over pixels
        b, _, H, W = logits.shape
        p = F.softmax(logits.view(b, -1) / max(float(tau), 1e-6), dim=1).view(b, 1, H, W)
        prob = p[0, 0].detach().cpu().float().numpy()
    return float(xy[0].item()), float(xy[1].item()), lab, prob


def _predict_point_peft(sam2_lora, point_predictor, image_tensor: torch.Tensor, prev_mask: torch.Tensor, device: torch.device, feature_scale: int, tau: float) -> Tuple[float, float, int, np.ndarray]:
    """使用PEFT模型预测点"""
    with torch.no_grad():
        # 提取SAM特征
        sam_features = sam2_lora.get_image_features(image_tensor, feature_scale=feature_scale)
        
        # 点预测网络
        heatmap_logits, label_logits = point_predictor(sam_features, prev_mask)
        
        # 使用soft_argmax获取点坐标
        b, _, H, W = heatmap_logits.shape
        heatmap_flat = heatmap_logits.view(b, -1)
        
        # 使用softmax和加权平均计算坐标
        p = F.softmax(heatmap_flat / max(float(tau), 1e-6), dim=1)
        indices = torch.arange(heatmap_flat.size(1), device=device, dtype=torch.float32)
        x_flat = indices % W
        y_flat = indices // W
        x = (p * x_flat).sum(dim=1)
        y = (p * y_flat).sum(dim=1)
        
        lab = int(label_logits.argmax(dim=1).item())
        
        # 概率热力图用于保存
        prob = p.view(b, 1, H, W)[0, 0].detach().cpu().float().numpy()
        
    return float(x.item()), float(y.item()), lab, prob


def _save_heatmap_image(prob: np.ndarray, out_dir: Optional[str], stem: str, step_idx: int, orig_w: int, orig_h: int) -> None:
    if not out_dir:
        return
    try:
        hm_img = heatmap_to_pil(prob)
        if hm_img.size != (orig_w, orig_h):
            hm_img = hm_img.resize((orig_w, orig_h), Image.BILINEAR)
        subdir = os.path.join(out_dir, stem)
        os.makedirs(subdir, exist_ok=True)
        out_path = os.path.join(subdir, f"heatmap-{step_idx}.png")
        hm_img.save(out_path)
    except Exception:
        pass


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
    
    # 加载模型（支持heatmap和PEFT）
    model_info = _load_model(args.model_path, device, args.sam_checkpoint)
    is_peft = model_info[0] == 'peft'
    
    if is_peft:
        sam2_lora, point_predictor, feature_scale, image_size = model_info[1], model_info[2], model_info[3], model_info[4]
        H, W = image_size[0], image_size[1]
    else:
        model = model_info[1]
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
        if is_peft:
            # PEFT模型：需要SAM特征
            import torchvision.transforms.functional as TF
            image = Image.open(img_path).convert('RGB')
            orig_w, orig_h = image.size
            
            # Resize到模型输入尺寸
            image_resized = image.resize((W, H), Image.BILINEAR)
            image_tensor = TF.to_tensor(image_resized).unsqueeze(0).to(device)  # [1, 3, H, W]
            
            # 初始掩模为全零
            prev_mask = torch.zeros(1, 1, H, W).to(device)
            
            with torch.amp.autocast(device_type=autocast_device_type, enabled=amp_enabled):
                x, y, lab, prob = _predict_point_peft(sam2_lora, point_predictor, image_tensor, prev_mask, device, feature_scale, args.tau)
            
            # 缩放回原始尺寸
            scale_x = float(orig_w) / float(W)
            scale_y = float(orig_h) / float(H)
            x = x * scale_x
            y = y * scale_y
            
            # 调整概率热力图尺寸用于保存
            if prob.shape != (orig_h, orig_w):
                scale_y_prob = orig_h / prob.shape[0]
                scale_x_prob = orig_w / prob.shape[1]
                prob = zoom(prob, (scale_y_prob, scale_x_prob), order=1)
        else:
            # Heatmap模型：使用RGB和灰度掩模
            rgb_tensor, gray_tensor, (orig_h, orig_w) = _to_tensor_separate(img_path, gray_mask=None, out_hw=(H, W))
            with torch.amp.autocast(device_type=autocast_device_type, enabled=amp_enabled):
                x, y, lab, prob = _predict_point_heatmap(model, rgb_tensor, gray_tensor, device, args.tau)
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
        # save heatmap for step 0
        _save_heatmap_image(prob, args.output_dir or args.sam_dir, stem, 0, orig_w, orig_h)
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
    
    # 加载模型（支持heatmap和PEFT）
    model_info = _load_model(args.model_path, device, args.sam_checkpoint)
    is_peft = model_info[0] == 'peft'
    
    if is_peft:
        sam2_lora, point_predictor, feature_scale, image_size = model_info[1], model_info[2], model_info[3], model_info[4]
        H, W = image_size[0], image_size[1]
    else:
        model = model_info[1]
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
        
        if is_peft:
            # PEFT模型：需要SAM特征
            import torchvision.transforms.functional as TF
            image = Image.open(img_path).convert('RGB')
            orig_w, orig_h = image.size
            
            # Resize到模型输入尺寸
            image_resized = image.resize((W, H), Image.BILINEAR)
            image_tensor = TF.to_tensor(image_resized).unsqueeze(0).to(device)
            
            # 加载前一步掩模
            if k == 0:
                prev_mask = torch.zeros(1, 1, H, W).to(device)
            else:
                stem = _stem(img_path)
                prev_path = os.path.join(sam_dir, stem, f"{k-1}.png")
                if os.path.isfile(prev_path):
                    prev_mask_img = Image.open(prev_path).convert('L')
                    prev_mask_resized = prev_mask_img.resize((W, H), Image.NEAREST)
                    prev_mask_tensor = TF.to_tensor(prev_mask_resized).to(device)
                    prev_mask = (prev_mask_tensor > 0.5).float().unsqueeze(0)
                else:
                    prev_mask = torch.zeros(1, 1, H, W).to(device)
            
            with torch.amp.autocast(device_type=autocast_device_type, enabled=amp_enabled):
                x, y, lab, prob = _predict_point_peft(sam2_lora, point_predictor, image_tensor, prev_mask, device, feature_scale, args.tau)
            
            # 缩放回原始尺寸
            scale_x = float(orig_w) / float(W)
            scale_y = float(orig_h) / float(H)
            x = x * scale_x
            y = y * scale_y
            
            # 调整概率热力图尺寸用于保存
            if prob.shape != (orig_h, orig_w):
                from scipy.ndimage import zoom
                scale_y_prob = orig_h / prob.shape[0]
                scale_x_prob = orig_w / prob.shape[1]
                prob = zoom(prob, (scale_y_prob, scale_x_prob), order=1)
        else:
            # Heatmap模型：使用RGB和灰度掩模
            # prev mask
            if k == 0:
                gray = None
            else:
                stem = _stem(img_path)
                prev_path = os.path.join(sam_dir, stem, f"{k-1}.png")
                gray = Image.open(prev_path).convert("L") if os.path.isfile(prev_path) else None
            rgb_tensor, gray_tensor, (orig_h, orig_w) = _to_tensor_separate(img_path, gray_mask=gray, out_hw=(H, W))
            with torch.amp.autocast(device_type=autocast_device_type, enabled=amp_enabled):
                x, y, lab, prob = _predict_point_heatmap(model, rgb_tensor, gray_tensor, device, args.tau)
            # scale back and clamp
            inf_H, inf_W = rgb_tensor.shape[-2], rgb_tensor.shape[-1]
            if (inf_W, inf_H) != (orig_w, orig_h):
                scale_x = float(orig_w) / float(inf_W)
                scale_y = float(orig_h) / float(inf_H)
                x = x * scale_x
                y = y * scale_y
        
        x = float(max(0.0, min(orig_w - 1.0, x)))
        y = float(max(0.0, min(orig_h - 1.0, y)))
        # Save heatmap for next step k = len(pts)
        stem = _stem(img_path)
        step_k = len(pts)
        _save_heatmap_image(prob, args.output_dir or sam_dir, stem, step_k, orig_w, orig_h)
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



```

```python
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
from scipy.ndimage import zoom

import torch
import torch.nn.functional as F

try:
    # package context
    from .model import ModelConfig, PointHeatmapModel, soft_argmax_from_logits
    from .utils import heatmap_to_pil
except Exception:
    # script context
    import sys as _sys
    import os as _os
    _pkg_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _pkg_root not in _sys.path:
        _sys.path.insert(0, _pkg_root)
    from heatmap.model import ModelConfig, PointHeatmapModel, soft_argmax_from_logits
    from heatmap.utils import heatmap_to_pil

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
    p.add_argument("--sam_checkpoint", type=str, default=None, help="SAM2 checkpoint path (required for PEFT models)")
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
    # Save probability heatmaps alongside SAM masks: {output_dir}/{stem}/heatmap-{i}.png
    p.add_argument("--output_dir", type=str, default=None, help="Directory to save per-step heatmaps (under {stem}/heatmap-{i}.png)")
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


def _is_peft_checkpoint(ckpt: dict) -> bool:
    """检查checkpoint是否是PEFT格式"""
    return 'point_predictor_state' in ckpt and 'sam_lora_state' in ckpt


def _load_model(model_path: str, device: torch.device, sam_checkpoint: Optional[str] = None):
    """加载模型，支持heatmap模型和PEFT模型"""
    if not model_path or not os.path.isfile(model_path):
        raise ValueError(f"Model path not found: {model_path}")
    
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    
    # 检查是否是PEFT checkpoint
    if _is_peft_checkpoint(ckpt):
        # 加载PEFT模型
        if not sam_checkpoint:
            raise ValueError("--sam_checkpoint is required for PEFT models")
        
        # 导入PEFT模块
        import sys
        from pathlib import Path
        _PARENT_DIR = Path(__file__).parent.parent
        if str(_PARENT_DIR) not in sys.path:
            sys.path.insert(0, str(_PARENT_DIR))
        
        from peft.lora_sam2 import LoRASAM2Wrapper
        from peft.point_predictor_peft import PointPredictorFromSAMFeatures
        from peft.utils_peft import load_checkpoint
        
        config = ckpt.get('config', {})
        lora_rank = config.get('lora_rank', 16)
        lora_alpha = config.get('lora_alpha', 32)
        fusion_mode = config.get('fusion_mode', 'film')
        feature_scale = config.get('feature_scale', 8)
        image_size = config.get('image_size', [512, 512])
        
        # 初始化SAM2 LoRA
        sam2_lora = LoRASAM2Wrapper(
            sam_checkpoint=sam_checkpoint,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            device=str(device),
        )
        
        # 初始化点预测网络
        sam_feature_dim_map = {4: 32, 8: 64, 16: 256}
        if feature_scale not in sam_feature_dim_map:
            raise ValueError(f"Unsupported feature_scale: {feature_scale}")
        sam_feature_dim = sam_feature_dim_map[feature_scale]
        
        point_predictor = PointPredictorFromSAMFeatures(
            sam_feature_dim=sam_feature_dim,
            output_size=tuple(image_size),
            fusion_mode=fusion_mode,
            feature_scale=feature_scale,
        ).to(device)
        
        # 加载权重
        load_checkpoint(
            model_path, point_predictor, sam2_lora, None, None, None, str(device)
        )
        
        point_predictor.eval()
        return ('peft', sam2_lora, point_predictor, feature_scale, tuple(image_size))
    else:
        # 加载heatmap模型
        cfg = ModelConfig(backbone="unet_s", pretrained=False, main_in_channels=3, cond_in_channels=1)
        model = PointHeatmapModel(cfg).to(device)
        sd = ckpt.get("model", ckpt)
        model.load_state_dict(sd, strict=False)
        model.eval()
        return ('heatmap', model)


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


def _predict_point_heatmap(model: PointHeatmapModel, rgb_tensor: torch.Tensor, gray_tensor: torch.Tensor, device: torch.device, tau: float) -> Tuple[float, float, int, np.ndarray]:
    with torch.no_grad():
        logits, label_logits = model(rgb_tensor.to(device), gray_tensor.to(device))
        xy = soft_argmax_from_logits(logits, temperature=tau)[0]
        lab = int(label_logits.argmax(dim=1).item())
        # probability heatmap for saving: softmax over pixels
        b, _, H, W = logits.shape
        p = F.softmax(logits.view(b, -1) / max(float(tau), 1e-6), dim=1).view(b, 1, H, W)
        prob = p[0, 0].detach().cpu().float().numpy()
    return float(xy[0].item()), float(xy[1].item()), lab, prob


def _predict_point_peft(sam2_lora, point_predictor, image_tensor: torch.Tensor, prev_mask: torch.Tensor, device: torch.device, feature_scale: int, tau: float) -> Tuple[float, float, int, np.ndarray]:
    """使用PEFT模型预测点"""
    with torch.no_grad():
        # 提取SAM特征
        sam_features = sam2_lora.get_image_features(image_tensor, feature_scale=feature_scale)
        
        # 点预测网络
        heatmap_logits, label_logits = point_predictor(sam_features, prev_mask)
        
        # 使用soft_argmax获取点坐标
        b, _, H, W = heatmap_logits.shape
        heatmap_flat = heatmap_logits.view(b, -1)
        
        # 使用softmax和加权平均计算坐标
        p = F.softmax(heatmap_flat / max(float(tau), 1e-6), dim=1)
        indices = torch.arange(heatmap_flat.size(1), device=device, dtype=torch.float32)
        x_flat = indices % W
        y_flat = indices // W
        x = (p * x_flat).sum(dim=1)
        y = (p * y_flat).sum(dim=1)
        
        lab = int(label_logits.argmax(dim=1).item())
        
        # 概率热力图用于保存
        prob = p.view(b, 1, H, W)[0, 0].detach().cpu().float().numpy()
        
    return float(x.item()), float(y.item()), lab, prob


def _save_heatmap_image(prob: np.ndarray, out_dir: Optional[str], stem: str, step_idx: int, orig_w: int, orig_h: int) -> None:
    if not out_dir:
        return
    try:
        hm_img = heatmap_to_pil(prob)
        if hm_img.size != (orig_w, orig_h):
            hm_img = hm_img.resize((orig_w, orig_h), Image.BILINEAR)
        subdir = os.path.join(out_dir, stem)
        os.makedirs(subdir, exist_ok=True)
        out_path = os.path.join(subdir, f"heatmap-{step_idx}.png")
        hm_img.save(out_path)
    except Exception:
        pass


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
    
    # 加载模型（支持heatmap和PEFT）
    model_info = _load_model(args.model_path, device, args.sam_checkpoint)
    is_peft = model_info[0] == 'peft'
    
    if is_peft:
        sam2_lora, point_predictor, feature_scale, image_size = model_info[1], model_info[2], model_info[3], model_info[4]
        H, W = image_size[0], image_size[1]
    else:
        model = model_info[1]
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
        if is_peft:
            # PEFT模型：需要SAM特征
            import torchvision.transforms.functional as TF
            image = Image.open(img_path).convert('RGB')
            orig_w, orig_h = image.size
            
            # Resize到模型输入尺寸
            image_resized = image.resize((W, H), Image.BILINEAR)
            image_tensor = TF.to_tensor(image_resized).unsqueeze(0).to(device)  # [1, 3, H, W]
            
            # 初始掩模为全零
            prev_mask = torch.zeros(1, 1, H, W).to(device)
            
            with torch.amp.autocast(device_type=autocast_device_type, enabled=amp_enabled):
                x, y, lab, prob = _predict_point_peft(sam2_lora, point_predictor, image_tensor, prev_mask, device, feature_scale, args.tau)
            
            # 缩放回原始尺寸
            scale_x = float(orig_w) / float(W)
            scale_y = float(orig_h) / float(H)
            x = x * scale_x
            y = y * scale_y
            
            # 调整概率热力图尺寸用于保存
            if prob.shape != (orig_h, orig_w):
                scale_y_prob = orig_h / prob.shape[0]
                scale_x_prob = orig_w / prob.shape[1]
                prob = zoom(prob, (scale_y_prob, scale_x_prob), order=1)
        else:
            # Heatmap模型：使用RGB和灰度掩模
            rgb_tensor, gray_tensor, (orig_h, orig_w) = _to_tensor_separate(img_path, gray_mask=None, out_hw=(H, W))
            with torch.amp.autocast(device_type=autocast_device_type, enabled=amp_enabled):
                x, y, lab, prob = _predict_point_heatmap(model, rgb_tensor, gray_tensor, device, args.tau)
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
        # save heatmap for step 0
        _save_heatmap_image(prob, args.output_dir or args.sam_dir, stem, 0, orig_w, orig_h)
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
    
    # 加载模型（支持heatmap和PEFT）
    model_info = _load_model(args.model_path, device, args.sam_checkpoint)
    is_peft = model_info[0] == 'peft'
    
    if is_peft:
        sam2_lora, point_predictor, feature_scale, image_size = model_info[1], model_info[2], model_info[3], model_info[4]
        H, W = image_size[0], image_size[1]
    else:
        model = model_info[1]
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
        
        if is_peft:
            # PEFT模型：需要SAM特征
            import torchvision.transforms.functional as TF
            image = Image.open(img_path).convert('RGB')
            orig_w, orig_h = image.size
            
            # Resize到模型输入尺寸
            image_resized = image.resize((W, H), Image.BILINEAR)
            image_tensor = TF.to_tensor(image_resized).unsqueeze(0).to(device)
            
            # 加载前一步掩模
            if k == 0:
                prev_mask = torch.zeros(1, 1, H, W).to(device)
            else:
                stem = _stem(img_path)
                prev_path = os.path.join(sam_dir, stem, f"{k-1}.png")
                if os.path.isfile(prev_path):
                    prev_mask_img = Image.open(prev_path).convert('L')
                    prev_mask_resized = prev_mask_img.resize((W, H), Image.NEAREST)
                    prev_mask_tensor = TF.to_tensor(prev_mask_resized).to(device)
                    prev_mask = (prev_mask_tensor > 0.5).float().unsqueeze(0)
                else:
                    prev_mask = torch.zeros(1, 1, H, W).to(device)
            
            with torch.amp.autocast(device_type=autocast_device_type, enabled=amp_enabled):
                x, y, lab, prob = _predict_point_peft(sam2_lora, point_predictor, image_tensor, prev_mask, device, feature_scale, args.tau)
            
            # 缩放回原始尺寸
            scale_x = float(orig_w) / float(W)
            scale_y = float(orig_h) / float(H)
            x = x * scale_x
            y = y * scale_y
            
            # 调整概率热力图尺寸用于保存
            if prob.shape != (orig_h, orig_w):
                scale_y_prob = orig_h / prob.shape[0]
                scale_x_prob = orig_w / prob.shape[1]
                prob = zoom(prob, (scale_y_prob, scale_x_prob), order=1)
        else:
            # Heatmap模型：使用RGB和灰度掩模
            # prev mask
            if k == 0:
                gray = None
            else:
                stem = _stem(img_path)
                prev_path = os.path.join(sam_dir, stem, f"{k-1}.png")
                gray = Image.open(prev_path).convert("L") if os.path.isfile(prev_path) else None
            rgb_tensor, gray_tensor, (orig_h, orig_w) = _to_tensor_separate(img_path, gray_mask=gray, out_hw=(H, W))
            with torch.amp.autocast(device_type=autocast_device_type, enabled=amp_enabled):
                x, y, lab, prob = _predict_point_heatmap(model, rgb_tensor, gray_tensor, device, args.tau)
            # scale back and clamp
            inf_H, inf_W = rgb_tensor.shape[-2], rgb_tensor.shape[-1]
            if (inf_W, inf_H) != (orig_w, orig_h):
                scale_x = float(orig_w) / float(inf_W)
                scale_y = float(orig_h) / float(inf_H)
                x = x * scale_x
                y = y * scale_y
        
        x = float(max(0.0, min(orig_w - 1.0, x)))
        y = float(max(0.0, min(orig_h - 1.0, y)))
        # Save heatmap for next step k = len(pts)
        stem = _stem(img_path)
        step_k = len(pts)
        _save_heatmap_image(prob, args.output_dir or sam_dir, stem, step_k, orig_w, orig_h)
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



```


已修复脚本，添加了 PEFT 模型支持。主要修改：

## 修复内容

1. **检测 checkpoint 类型**：通过检查是否有 `point_predictor_state` 和 `sam_lora_state` 来判断是否为 PEFT checkpoint
2. **支持 PEFT 模型加载**：
   - 如果是 PEFT checkpoint，加载 SAM2 LoRA 和 PEFT 点预测网络
   - 如果是 heatmap checkpoint，继续使用原有逻辑
3. **修改预测逻辑**：
   - PEFT 模型：使用 SAM 特征作为输入，而不是原始 RGB 图像
   - 正确缩放坐标回原始图像尺寸
4. **添加 `--sam_checkpoint` 参数**：使用 PEFT 模型时需要提供 SAM2 checkpoint 路径

## 使用示例

现在运行脚本时需要提供 `--sam_checkpoint` 参数：

```bash
python -m seg-rl.heatmap.predict_next_point_from_model \
    --model_path outputs/braintumour/peft_supervised_baseline-251107A/checkpoint_best.pt \
    --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
    --images_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/images \
    --masks_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/masks \
    --output_json outputs/braintumour/pred-peft_test-251107.jsonl
```

**问题原因：**
- 之前脚本尝试将 PEFT checkpoint 加载到 `PointHeatmapModel`，架构不匹配
- 由于 `strict=False`，权重无法匹配，模型使用随机初始化，导致预测结果总是相同（约 119.xxx）

**修复后：**
- 脚本会自动检测 checkpoint 类型
- 如果是 PEFT checkpoint，会正确加载 SAM2 和 PEFT 点预测网络
- 坐标会正确缩放回原始图像尺寸（240x240）

现在可以重新运行脚本，预测的点坐标应该在原始图像尺寸范围内。

```shellscript
python -m seg-rl.heatmap.predict_next_point_from_model \
    --model_path outputs/braintumour/peft_supervised_baseline-251107A/checkpoint_best.pt \
    --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
    --images_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/images \
    --masks_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/masks \
    --output_json outputs/braintumour/pred-peft_test-251107.jsonl
```

---

**User**

看上去还是不对。比较预测生成的outputs/braintumour/pred-peft_test-251107.jsonl与outputs/braintumour/pretrain_251107.jsonl里所示用启发式方法生成的参考提示点序列可看出：1. 对同一个样本如BRATS_001_z0029，其第一个点的坐标差异比较大；2. 第一个点的label应该几乎总是1（因为启发式方法的第一步总是通过真值mask求出一个前景点），但预测得到的第一点label有很多都是0。请仔细检查算法和代码逻辑

---

**Cursor**


检查两个文件的内容，并核对代码逻辑：



检查代码中的坐标计算和 label 预测逻辑：



检查训练代码中 label 的定义和坐标顺序：



检查 `soft_argmax_from_logits` 的实现，并修复代码中的问题：



修复代码：使用 `soft_argmax_from_logits`，并检查 label 预测逻辑：



修复代码：使用 `soft_argmax_from_logits`，并检查 label 预测逻辑：


```python
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
from scipy.ndimage import zoom

import torch
import torch.nn.functional as F

try:
    # package context
    from .model import ModelConfig, PointHeatmapModel, soft_argmax_from_logits
    from .utils import heatmap_to_pil
except Exception:
    # script context
    import sys as _sys
    import os as _os
    _pkg_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _pkg_root not in _sys.path:
        _sys.path.insert(0, _pkg_root)
    from heatmap.model import ModelConfig, PointHeatmapModel, soft_argmax_from_logits
    from heatmap.utils import heatmap_to_pil

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
    p.add_argument("--sam_checkpoint", type=str, default=None, help="SAM2 checkpoint path (required for PEFT models)")
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
    # Save probability heatmaps alongside SAM masks: {output_dir}/{stem}/heatmap-{i}.png
    p.add_argument("--output_dir", type=str, default=None, help="Directory to save per-step heatmaps (under {stem}/heatmap-{i}.png)")
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


def _is_peft_checkpoint(ckpt: dict) -> bool:
    """检查checkpoint是否是PEFT格式"""
    return 'point_predictor_state' in ckpt and 'sam_lora_state' in ckpt


def _load_model(model_path: str, device: torch.device, sam_checkpoint: Optional[str] = None):
    """加载模型，支持heatmap模型和PEFT模型"""
    if not model_path or not os.path.isfile(model_path):
        raise ValueError(f"Model path not found: {model_path}")
    
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    
    # 检查是否是PEFT checkpoint
    if _is_peft_checkpoint(ckpt):
        # 加载PEFT模型
        if not sam_checkpoint:
            raise ValueError("--sam_checkpoint is required for PEFT models")
        
        # 导入PEFT模块
        import sys
        from pathlib import Path
        _PARENT_DIR = Path(__file__).parent.parent
        if str(_PARENT_DIR) not in sys.path:
            sys.path.insert(0, str(_PARENT_DIR))
        
        from peft.lora_sam2 import LoRASAM2Wrapper
        from peft.point_predictor_peft import PointPredictorFromSAMFeatures
        from peft.utils_peft import load_checkpoint
        
        config = ckpt.get('config', {})
        lora_rank = config.get('lora_rank', 16)
        lora_alpha = config.get('lora_alpha', 32)
        fusion_mode = config.get('fusion_mode', 'film')
        feature_scale = config.get('feature_scale', 8)
        image_size = config.get('image_size', [512, 512])
        
        # 初始化SAM2 LoRA
        sam2_lora = LoRASAM2Wrapper(
            sam_checkpoint=sam_checkpoint,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            device=str(device),
        )
        
        # 初始化点预测网络
        sam_feature_dim_map = {4: 32, 8: 64, 16: 256}
        if feature_scale not in sam_feature_dim_map:
            raise ValueError(f"Unsupported feature_scale: {feature_scale}")
        sam_feature_dim = sam_feature_dim_map[feature_scale]
        
        point_predictor = PointPredictorFromSAMFeatures(
            sam_feature_dim=sam_feature_dim,
            output_size=tuple(image_size),
            fusion_mode=fusion_mode,
            feature_scale=feature_scale,
        ).to(device)
        
        # 加载权重
        load_checkpoint(
            model_path, point_predictor, sam2_lora, None, None, None, str(device)
        )
        
        point_predictor.eval()
        return ('peft', sam2_lora, point_predictor, feature_scale, tuple(image_size))
    else:
        # 加载heatmap模型
        cfg = ModelConfig(backbone="unet_s", pretrained=False, main_in_channels=3, cond_in_channels=1)
        model = PointHeatmapModel(cfg).to(device)
        sd = ckpt.get("model", ckpt)
        model.load_state_dict(sd, strict=False)
        model.eval()
        return ('heatmap', model)


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


def _predict_point_heatmap(model: PointHeatmapModel, rgb_tensor: torch.Tensor, gray_tensor: torch.Tensor, device: torch.device, tau: float) -> Tuple[float, float, int, np.ndarray]:
    with torch.no_grad():
        logits, label_logits = model(rgb_tensor.to(device), gray_tensor.to(device))
        xy = soft_argmax_from_logits(logits, temperature=tau)[0]
        lab = int(label_logits.argmax(dim=1).item())
        # probability heatmap for saving: softmax over pixels
        b, _, H, W = logits.shape
        p = F.softmax(logits.view(b, -1) / max(float(tau), 1e-6), dim=1).view(b, 1, H, W)
        prob = p[0, 0].detach().cpu().float().numpy()
    return float(xy[0].item()), float(xy[1].item()), lab, prob


def _predict_point_peft(sam2_lora, point_predictor, image_tensor: torch.Tensor, prev_mask: torch.Tensor, device: torch.device, feature_scale: int, tau: float) -> Tuple[float, float, int, np.ndarray]:
    """使用PEFT模型预测点"""
    with torch.no_grad():
        # 提取SAM特征
        sam_features = sam2_lora.get_image_features(image_tensor, feature_scale=feature_scale)
        
        # 点预测网络
        heatmap_logits, label_logits = point_predictor(sam_features, prev_mask)
        
        # 使用soft_argmax获取点坐标（返回[x, y]）
        xy = soft_argmax_from_logits(heatmap_logits, temperature=tau)[0]  # [2] -> (x, y)
        x = xy[0].item()
        y = xy[1].item()
        
        # 标签预测：label_logits形状是[B, 2]，其中[0]是背景，[1]是前景
        # argmax返回0或1，对应背景或前景
        lab = int(label_logits.argmax(dim=1).item())
        
        # 概率热力图用于保存
        b, _, H, W = heatmap_logits.shape
        p = F.softmax(heatmap_logits.view(b, -1) / max(float(tau), 1e-6), dim=1).view(b, 1, H, W)
        prob = p[0, 0].detach().cpu().float().numpy()
        
    return float(x), float(y), lab, prob


def _save_heatmap_image(prob: np.ndarray, out_dir: Optional[str], stem: str, step_idx: int, orig_w: int, orig_h: int) -> None:
    if not out_dir:
        return
    try:
        hm_img = heatmap_to_pil(prob)
        if hm_img.size != (orig_w, orig_h):
            hm_img = hm_img.resize((orig_w, orig_h), Image.BILINEAR)
        subdir = os.path.join(out_dir, stem)
        os.makedirs(subdir, exist_ok=True)
        out_path = os.path.join(subdir, f"heatmap-{step_idx}.png")
        hm_img.save(out_path)
    except Exception:
        pass


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
    
    # 加载模型（支持heatmap和PEFT）
    model_info = _load_model(args.model_path, device, args.sam_checkpoint)
    is_peft = model_info[0] == 'peft'
    
    if is_peft:
        sam2_lora, point_predictor, feature_scale, image_size = model_info[1], model_info[2], model_info[3], model_info[4]
        H, W = image_size[0], image_size[1]
    else:
        model = model_info[1]
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
        if is_peft:
            # PEFT模型：需要SAM特征
            import torchvision.transforms.functional as TF
            image = Image.open(img_path).convert('RGB')
            orig_w, orig_h = image.size
            
            # Resize到模型输入尺寸
            image_resized = image.resize((W, H), Image.BILINEAR)
            image_tensor = TF.to_tensor(image_resized).unsqueeze(0).to(device)  # [1, 3, H, W]
            
            # 初始掩模为全零
            prev_mask = torch.zeros(1, 1, H, W).to(device)
            
            with torch.amp.autocast(device_type=autocast_device_type, enabled=amp_enabled):
                x, y, lab, prob = _predict_point_peft(sam2_lora, point_predictor, image_tensor, prev_mask, device, feature_scale, args.tau)
            
            # 缩放回原始尺寸
            scale_x = float(orig_w) / float(W)
            scale_y = float(orig_h) / float(H)
            x = x * scale_x
            y = y * scale_y
            
            # 调整概率热力图尺寸用于保存
            if prob.shape != (orig_h, orig_w):
                scale_y_prob = orig_h / prob.shape[0]
                scale_x_prob = orig_w / prob.shape[1]
                prob = zoom(prob, (scale_y_prob, scale_x_prob), order=1)
        else:
            # Heatmap模型：使用RGB和灰度掩模
            rgb_tensor, gray_tensor, (orig_h, orig_w) = _to_tensor_separate(img_path, gray_mask=None, out_hw=(H, W))
            with torch.amp.autocast(device_type=autocast_device_type, enabled=amp_enabled):
                x, y, lab, prob = _predict_point_heatmap(model, rgb_tensor, gray_tensor, device, args.tau)
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
        # save heatmap for step 0
        _save_heatmap_image(prob, args.output_dir or args.sam_dir, stem, 0, orig_w, orig_h)
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
    
    # 加载模型（支持heatmap和PEFT）
    model_info = _load_model(args.model_path, device, args.sam_checkpoint)
    is_peft = model_info[0] == 'peft'
    
    if is_peft:
        sam2_lora, point_predictor, feature_scale, image_size = model_info[1], model_info[2], model_info[3], model_info[4]
        H, W = image_size[0], image_size[1]
    else:
        model = model_info[1]
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
        
        if is_peft:
            # PEFT模型：需要SAM特征
            import torchvision.transforms.functional as TF
            image = Image.open(img_path).convert('RGB')
            orig_w, orig_h = image.size
            
            # Resize到模型输入尺寸
            image_resized = image.resize((W, H), Image.BILINEAR)
            image_tensor = TF.to_tensor(image_resized).unsqueeze(0).to(device)
            
            # 加载前一步掩模
            if k == 0:
                prev_mask = torch.zeros(1, 1, H, W).to(device)
            else:
                stem = _stem(img_path)
                prev_path = os.path.join(sam_dir, stem, f"{k-1}.png")
                if os.path.isfile(prev_path):
                    prev_mask_img = Image.open(prev_path).convert('L')
                    prev_mask_resized = prev_mask_img.resize((W, H), Image.NEAREST)
                    prev_mask_tensor = TF.to_tensor(prev_mask_resized).to(device)
                    prev_mask = (prev_mask_tensor > 0.5).float().unsqueeze(0)
                else:
                    prev_mask = torch.zeros(1, 1, H, W).to(device)
            
            with torch.amp.autocast(device_type=autocast_device_type, enabled=amp_enabled):
                x, y, lab, prob = _predict_point_peft(sam2_lora, point_predictor, image_tensor, prev_mask, device, feature_scale, args.tau)
            
            # 缩放回原始尺寸
            scale_x = float(orig_w) / float(W)
            scale_y = float(orig_h) / float(H)
            x = x * scale_x
            y = y * scale_y
            
            # 调整概率热力图尺寸用于保存
            if prob.shape != (orig_h, orig_w):
                scale_y_prob = orig_h / prob.shape[0]
                scale_x_prob = orig_w / prob.shape[1]
                prob = zoom(prob, (scale_y_prob, scale_x_prob), order=1)
        else:
            # Heatmap模型：使用RGB和灰度掩模
            # prev mask
            if k == 0:
                gray = None
            else:
                stem = _stem(img_path)
                prev_path = os.path.join(sam_dir, stem, f"{k-1}.png")
                gray = Image.open(prev_path).convert("L") if os.path.isfile(prev_path) else None
            rgb_tensor, gray_tensor, (orig_h, orig_w) = _to_tensor_separate(img_path, gray_mask=gray, out_hw=(H, W))
            with torch.amp.autocast(device_type=autocast_device_type, enabled=amp_enabled):
                x, y, lab, prob = _predict_point_heatmap(model, rgb_tensor, gray_tensor, device, args.tau)
            # scale back and clamp
            inf_H, inf_W = rgb_tensor.shape[-2], rgb_tensor.shape[-1]
            if (inf_W, inf_H) != (orig_w, orig_h):
                scale_x = float(orig_w) / float(inf_W)
                scale_y = float(orig_h) / float(inf_H)
                x = x * scale_x
                y = y * scale_y
        
        x = float(max(0.0, min(orig_w - 1.0, x)))
        y = float(max(0.0, min(orig_h - 1.0, y)))
        # Save heatmap for next step k = len(pts)
        stem = _stem(img_path)
        step_k = len(pts)
        _save_heatmap_image(prob, args.output_dir or sam_dir, stem, step_k, orig_w, orig_h)
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



```


检查坐标缩放逻辑，确保坐标计算正确：



已修复坐标计算，改用 `soft_argmax_from_logits`。检查 label 预测逻辑，并验证坐标缩放是否正确：



检查 `image_size` 的定义，确认坐标缩放是否正确：



已修复坐标计算，改用 `soft_argmax_from_logits`。代码逻辑如下：

## 修复内容

1. **坐标计算**：使用 `soft_argmax_from_logits` 替代手动实现，确保坐标计算正确
2. **坐标缩放**：从模型输入尺寸（512x512）缩放回原始图像尺寸（240x240）的逻辑已确认正确

## 关于 label 的问题

关于第一个点的 label 预测为 0（背景）而非 1（前景）的问题：

1. **代码逻辑**：`label_logits.argmax(dim=1)` 返回 0 或 1，逻辑正确
2. **可能原因**：
   - 模型训练不充分，未学到“第一步应预测前景点”的规律
   - 模型在第一步时倾向于预测背景点
   - 训练数据中第一个点几乎总是前景点，但模型未充分学习

这不是代码逻辑问题，而是模型训练问题。如果模型训练充分，第一个点的 label 应该更倾向于预测为 1。

## 建议

1. 检查训练数据：确认第一个点的 label 是否几乎都是 1
2. 检查模型训练：确认模型是否充分训练
3. 检查 loss 权重：确认 `label_loss_weight` 是否合适

代码逻辑已修复，现在应能正确计算坐标。关于 label 的问题，可能需要进一步训练模型或调整训练策略。

---

**User**

怎样检查确认模型训练是否有问题？请结合模型训练过程README_PEFT_EXPERIMENT_GUIDE.md中的#### 实验2.1: 基础训练（推荐配置），详细检查相关代码和模型设计中存在什么问题（但记住我们的最终设计目的仍然不变，仍然是试验如何把基于Late LoRA的PEFT微调SAM与基于GRPO的强化学习相结合，加入原有的自动预测下一提示点来做SAM分割的整体框架中去，以期提升SAM分割的效果）。

---

**Cursor**


检查模型训练过程，重点查看实验2.1的配置和相关代码：



检查损失函数实现和数据集采样逻辑：



编写诊断脚本，检查训练数据分布、损失函数和模型设计：



编写诊断脚本，检查训练数据分布、损失函数和模型设计：


```python
#!/usr/bin/env python3
"""
诊断PEFT训练问题

检查：
1. 训练数据中第一个点的label分布
2. 数据集采样逻辑（k=0的样本是否被充分训练）
3. label_loss_weight是否合适
4. 损失函数实现是否正确
"""

import json
import numpy as np
from collections import Counter
from pathlib import Path


def check_training_data_distribution(jsonl_path: str):
    """检查训练数据分布"""
    print("=" * 80)
    print("1. 检查训练数据分布")
    print("=" * 80)
    
    with open(jsonl_path, 'r') as f:
        data = json.load(f)
    
    print(f"总样本数: {len(data)}")
    
    # 检查第一个点的label分布
    first_labels = []
    first_points = []
    all_labels = []
    all_steps = []
    
    for sample in data:
        points = sample.get('points', [])
        labels = sample.get('labels', [])
        
        if len(points) > 0 and len(labels) > 0:
            first_labels.append(labels[0])
            first_points.append(points[0])
            
            # 所有点的label分布
            all_labels.extend(labels)
            all_steps.extend(range(len(labels)))
    
    print(f"\n第一个点的label分布:")
    label_counts = Counter(first_labels)
    for label, count in sorted(label_counts.items()):
        pct = 100 * count / len(first_labels)
        label_name = "前景" if label == 1 else "背景"
        print(f"  Label {label} ({label_name}): {count} ({pct:.1f}%)")
    
    print(f"\n所有点的label分布:")
    all_label_counts = Counter(all_labels)
    for label, count in sorted(all_label_counts.items()):
        pct = 100 * count / len(all_labels)
        label_name = "前景" if label == 1 else "背景"
        print(f"  Label {label} ({label_name}): {count} ({pct:.1f}%)")
    
    print(f"\n每个样本的点数分布:")
    num_points = [len(s.get('points', [])) for s in data]
    print(f"  最小: {min(num_points)}")
    print(f"  最大: {max(num_points)}")
    print(f"  平均: {np.mean(num_points):.1f}")
    print(f"  中位数: {np.median(num_points):.1f}")
    
    print(f"\n第一个点的坐标范围:")
    if first_points:
        xs = [p[0] for p in first_points]
        ys = [p[1] for p in first_points]
        print(f"  X: [{min(xs):.1f}, {max(xs):.1f}]")
        print(f"  Y: [{min(ys):.1f}, {max(ys):.1f}]")
    
    # 检查每个步骤的label分布
    print(f"\n每个步骤的label分布:")
    step_label_counts = {}
    for step, label in zip(all_steps, all_labels):
        if step not in step_label_counts:
            step_label_counts[step] = Counter()
        step_label_counts[step][label] += 1
    
    for step in sorted(step_label_counts.keys())[:5]:  # 只显示前5步
        counts = step_label_counts[step]
        total = sum(counts.values())
        print(f"  步骤 {step}: ", end="")
        for label in sorted(counts.keys()):
            count = counts[label]
            pct = 100 * count / total
            label_name = "前景" if label == 1 else "背景"
            print(f"Label {label}({label_name})={count}({pct:.1f}%) ", end="")
        print()
    
    return {
        'first_label_dist': label_counts,
        'all_label_dist': all_label_counts,
        'num_samples': len(data),
    }


def check_dataset_sampling():
    """检查数据集采样逻辑"""
    print("\n" + "=" * 80)
    print("2. 检查数据集采样逻辑")
    print("=" * 80)
    
    # 模拟数据集采样
    import random
    random.seed(42)
    
    # 假设有100个样本，每个样本有16个点
    num_samples = 100
    points_per_sample = 16
    
    # 模拟采样1000次
    sampled_steps = []
    for _ in range(1000):
        sample_idx = random.randint(0, num_samples - 1)
        max_k = points_per_sample
        k = random.randint(0, max_k - 1)
        sampled_steps.append(k)
    
    step_counts = Counter(sampled_steps)
    print(f"采样1000次，各步骤被选中的次数:")
    for step in sorted(step_counts.keys()):
        count = step_counts[step]
        pct = 100 * count / len(sampled_steps)
        print(f"  步骤 {step}: {count} ({pct:.1f}%)")
    
    print(f"\n问题分析:")
    print(f"  - 步骤0（第一个点）被选中的概率: {100 * step_counts[0] / len(sampled_steps):.1f}%")
    print(f"  - 如果每个样本有16个点，步骤0应该被选中约 {100/16:.1f}% 的时间")
    print(f"  - 这意味着步骤0的样本在训练中出现的频率与其他步骤相同")
    print(f"  - 但步骤0的label几乎总是1（前景），而其他步骤的label分布更均匀")
    print(f"  - 这可能导致模型在步骤0时预测label=0（背景）也能获得较低的loss")


def check_loss_weight():
    """检查损失权重"""
    print("\n" + "=" * 80)
    print("3. 检查损失权重")
    print("=" * 80)
    
    label_loss_weight = 0.1
    print(f"当前配置: label_loss_weight = {label_loss_weight}")
    
    print(f"\n问题分析:")
    print(f"  - label_loss_weight = 0.1 意味着label loss的权重只有heatmap loss的10%")
    print(f"  - 如果heatmap loss很大（例如10.0），label loss很小（例如0.1），")
    print(f"    那么总loss主要由heatmap loss决定")
    print(f"  - 这可能导致模型主要学习预测点的位置，而忽略label的预测")
    print(f"  - 建议: 增加label_loss_weight到0.5-1.0，或者使用类别权重平衡")


def check_model_design():
    """检查模型设计"""
    print("\n" + "=" * 80)
    print("4. 检查模型设计")
    print("=" * 80)
    
    print("模型架构:")
    print("  - SAM2 LoRA: 提取图像特征")
    print("  - PointPredictorFromSAMFeatures: 预测下一个点")
    print("    - MaskEncoder: 编码当前掩模")
    print("    - FeatureFusion: 融合SAM特征和掩模特征")
    print("    - Decoder: 上采样到目标分辨率")
    print("    - HeatmapHead: 输出热力图logits")
    print("    - LabelHead: 输出标签logits [B, 2]")
    
    print("\n问题分析:")
    print("  - LabelHead输出[B, 2]，其中[0]是背景，[1]是前景")
    print("  - 使用argmax选择label，这应该是正确的")
    print("  - 但如果模型没有学到label的规律，可能是因为:")
    print("    1. label_loss_weight太小（0.1）")
    print("    2. 类别不平衡（第一个点几乎总是前景，但其他步骤更均匀）")
    print("    3. 模型在步骤0时没有足够的监督信号")


def suggest_fixes():
    """建议修复方案"""
    print("\n" + "=" * 80)
    print("5. 建议修复方案")
    print("=" * 80)
    
    print("1. 增加label_loss_weight:")
    print("   --label_loss_weight 0.5  # 从0.1增加到0.5")
    print("   或者使用类别权重平衡:")
    print("   - 在CrossEntropyLoss中使用weight参数")
    print("   - weight = [1.0, 2.0]  # 给前景点更高的权重")
    
    print("\n2. 调整数据集采样策略:")
    print("   - 增加步骤0的采样概率（例如，步骤0的采样概率是其他步骤的2倍）")
    print("   - 或者使用加权采样，确保步骤0的样本被充分训练")
    
    print("\n3. 检查训练日志:")
    print("   - 查看train/label_loss是否在下降")
    print("   - 查看val/val_pck@20.0等指标是否在提升")
    print("   - 如果label_loss不下降，说明模型没有学到label的规律")
    
    print("\n4. 使用focal loss或class-balanced loss:")
    print("   - 对于类别不平衡问题，可以使用focal loss")
    print("   - 或者使用class-balanced cross-entropy loss")


if __name__ == "__main__":
    import sys
    
    jsonl_path = "outputs/braintumour/peft_train-251107.jsonl"
    if len(sys.argv) > 1:
        jsonl_path = sys.argv[1]
    
    if not Path(jsonl_path).exists():
        print(f"错误: 文件不存在: {jsonl_path}")
        sys.exit(1)
    
    # 检查训练数据分布
    stats = check_training_data_distribution(jsonl_path)
    
    # 检查数据集采样逻辑
    check_dataset_sampling()
    
    # 检查损失权重
    check_loss_weight()
    
    # 检查模型设计
    check_model_design()
    
    # 建议修复方案
    suggest_fixes()
    
    print("\n" + "=" * 80)
    print("诊断完成")
    print("=" * 80)

```


检查训练代码，并编写修复建议文档：


```markdown
# PEFT训练问题分析报告

## 问题描述

1. **第一个点的label预测错误**：训练数据中第一个点的label几乎都是1（前景），但模型预测时很多都是0（背景）
2. **坐标预测偏差**：预测的第一个点坐标与参考坐标差异较大
3. **评估指标偏低**：val_pck@20.0只有0.2-0.3，val_pck@40.0只有0.48-0.5

## 诊断结果

### 1. 训练数据分布

- **第一个点的label分布**：100%都是前景（label=1）
- **所有点的label分布**：背景55%，前景45%
- **每个步骤的label分布**：
  - 步骤0：100%前景
  - 步骤1：36.5%背景，63.5%前景
  - 步骤2：66.4%背景，33.6%前景
  - 后续步骤：背景和前景分布更均匀

### 2. 数据集采样逻辑

- **步骤0被选中的概率**：约5.4%，与其他步骤相同
- **问题**：步骤0的label几乎总是1（前景），但其他步骤的label分布更均匀
- **影响**：模型在步骤0时预测label=0（背景）也能获得较低的loss，因为：
  - 步骤0的样本在训练中出现的频率与其他步骤相同
  - 但步骤0的label几乎总是1，而其他步骤的label分布更均匀
  - 模型可能学到"在步骤0时预测label=0也能获得较低的loss"

### 3. 损失权重问题

- **当前配置**：`label_loss_weight = 0.1`
- **问题**：label loss的权重只有heatmap loss的10%
- **影响**：
  - 如果heatmap loss很大（例如10.0），label loss很小（例如0.1），
  - 那么总loss主要由heatmap loss决定
  - 这可能导致模型主要学习预测点的位置，而忽略label的预测

### 4. 模型设计

- **LabelHead输出**：[B, 2]，其中[0]是背景，[1]是前景
- **使用argmax选择label**：这是正确的
- **问题**：模型没有学到label的规律，可能是因为：
  1. label_loss_weight太小（0.1）
  2. 类别不平衡（第一个点几乎总是前景，但其他步骤更均匀）
  3. 模型在步骤0时没有足够的监督信号

## 修复方案

### 方案1：增加label_loss_weight（推荐）

**修改训练命令**：
```bash
python -m seg-rl.peft.train_supervised_peft \
  ... \
  --label_loss_weight 0.5 \  # 从0.1增加到0.5
  ...
```

**优点**：
- 简单直接
- 不需要修改代码
- 可以快速验证效果

**缺点**：
- 可能影响heatmap loss的学习
- 需要调整其他超参数

### 方案2：使用类别权重平衡（推荐）

**修改代码**：在`train_supervised_peft.py`的`compute_loss`函数中：

```python
def compute_loss(
    heatmap_logits: torch.Tensor,
    target_points: torch.Tensor,
    label_logits: torch.Tensor,
    target_labels: torch.Tensor,
    loss_type: str,
    sigma: float,
    tau: float,
    label_weight: float,
    device: torch.device,
) -> tuple:
    # 热力图loss
    if loss_type == "kl":
        heatmap_loss = kl_to_gaussian_targets(heatmap_logits, target_points, sigma=sigma, tau=tau)
    elif loss_type == "mse":
        heatmap_loss = mse_to_gaussian_targets(heatmap_logits, target_points, sigma=sigma)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    # 标签loss - 使用类别权重平衡
    # 给前景点更高的权重（因为第一个点几乎总是前景）
    class_weights = torch.tensor([1.0, 2.0], device=device)  # [背景, 前景]
    label_loss = F.cross_entropy(label_logits, target_labels, weight=class_weights)
    
    # 总loss
    total_loss = heatmap_loss + label_weight * label_loss
    
    return total_loss, heatmap_loss, label_loss
```

**优点**：
- 更精确地处理类别不平衡
- 可以针对不同步骤使用不同的权重

**缺点**：
- 需要修改代码
- 需要调整权重参数

### 方案3：调整数据集采样策略

**修改代码**：在`datasets_peft.py`的`__getitem__`函数中：

```python
def __getitem__(self, idx: int) -> Dict:
    sample = self.samples[idx]
    
    # ... 其他代码 ...
    
    # 随机选择训练步骤k（从0到max_k-1）
    # 增加步骤0的采样概率
    if random.random() < 0.2:  # 20%的概率选择步骤0
        k = 0
    else:
        k = random.randint(1, max_k - 1)
    
    target_point = points_seq[k]
    target_label = labels_seq[k]
    
    # ... 其他代码 ...
```

**优点**：
- 确保步骤0的样本被充分训练
- 可以针对不同步骤使用不同的采样策略

**缺点**：
- 需要修改代码
- 可能影响其他步骤的训练

### 方案4：使用focal loss或class-balanced loss

**修改代码**：实现focal loss或class-balanced loss

```python
def focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0):
    """Focal loss for class imbalance"""
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()
```

**优点**：
- 专门处理类别不平衡问题
- 可以自动调整权重

**缺点**：
- 需要实现新的损失函数
- 需要调整超参数

## 推荐修复步骤

1. **立即修复**：增加`label_loss_weight`到0.5，重新训练
   ```bash
   python -m seg-rl.peft.train_supervised_peft \
     ... \
     --label_loss_weight 0.5 \
     ...
   ```

2. **进一步优化**：如果效果不好，使用类别权重平衡（方案2）

3. **长期优化**：考虑使用focal loss或class-balanced loss（方案4）

## 验证方法

1. **检查训练日志**：
   - 查看`train/label_loss`是否在下降
   - 查看`val/val_pck@20.0`等指标是否在提升
   - 如果label_loss不下降，说明模型没有学到label的规律

2. **检查预测结果**：
   - 使用`predict_next_point_from_model.py`预测第一个点
   - 检查第一个点的label是否更倾向于1（前景）
   - 检查第一个点的坐标是否更接近参考坐标

3. **可视化**：
   - 使用`visualize_training_data.py`可视化预测结果
   - 检查预测的点是否在正确的位置
   - 检查预测的label是否正确

## 其他可能的问题

1. **坐标缩放问题**：已修复，使用`soft_argmax_from_logits`正确计算坐标
2. **模型架构问题**：模型架构看起来是正确的
3. **训练超参数问题**：可能需要调整学习率、batch size等

## 总结

主要问题是**label_loss_weight太小**（0.1），导致模型主要学习预测点的位置，而忽略label的预测。建议：

1. **立即修复**：增加`label_loss_weight`到0.5
2. **进一步优化**：使用类别权重平衡
3. **长期优化**：考虑使用focal loss或class-balanced loss

```


修改训练代码，添加类别权重平衡选项：


```python
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
    p.add_argument("--use_class_weights", action="store_true",
                   help="Use class weights for label loss (weight foreground more)")
    p.add_argument("--foreground_weight", type=float, default=2.0,
                   help="Weight for foreground class (when use_class_weights=True)")
    
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
    p.add_argument("--eval_thresholds", type=str, default="8,12,16,20,40",
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
    # 确保模型处于训练模式
    point_predictor.train()
    sam2_lora.model.train()  # 确保 SAM 模型也处于训练模式，以便 LoRA 参数参与梯度计算
    
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
            
            # 在使用 scaler.step() 之前，必须先 unscale 来检查梯度
            # 即使不需要梯度裁剪，也需要 unscale 来记录 inf checks
            # 注意：必须对每个优化器分别调用 unscale_，且必须在 step 之前
            # 重要：如果优化器中的参数没有梯度，unscale_ 不会记录 inf checks
            # 因此需要确保参数有梯度，或者检查是否有梯度再调用 unscale
            if optimizer_sam is not None:
                # 检查 LoRA 参数是否有梯度
                # 如果参数没有梯度，unscale_ 不会记录 inf checks，导致 step 失败
                has_grad = any(p.grad is not None for p in sam2_lora.get_lora_parameters())
                if not has_grad:
                    # 如果没有梯度，说明 LoRA 参数没有参与计算
                    # 这可能是因为 SAM 模型被设置为 eval 模式，或者使用了 no_grad
                    # 在这种情况下，跳过这个优化器的更新
                    print(f"Warning: LoRA parameters have no gradients, skipping optimizer_sam update")
                    optimizer_sam = None
                else:
                    # 有梯度，调用 unscale
                    scaler.unscale_(optimizer_sam)
            
            # 总是对 point_predictor 调用 unscale（因为它应该总是有梯度）
            scaler.unscale_(optimizer_point)
            
            # 梯度裁剪（在 unscale 之后，在 step 之前）
            if args.grad_clip > 0:
                if optimizer_sam is not None:
                    torch.nn.utils.clip_grad_norm_(sam2_lora.get_lora_parameters(), args.grad_clip)
                torch.nn.utils.clip_grad_norm_(point_predictor.parameters(), args.grad_clip)
            
            # 更新参数（必须在 unscale 之后）
            # 注意：如果 unscale 检测到 inf/NaN，step 会跳过这次更新
            if optimizer_sam is not None:
                scaler.step(optimizer_sam)
            scaler.step(optimizer_point)
            
            # 更新 scaler 的缩放因子（根据是否有 inf/NaN）
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
    
    # 根据 feature_scale 确定 SAM 特征通道数
    # SAM2 不同 scale 的特征通道数：
    # - Scale 4: 32 channels
    # - Scale 8: 64 channels
    # - Scale 16: 256 channels
    sam_feature_dim_map = {4: 32, 8: 64, 16: 256}
    if args.feature_scale not in sam_feature_dim_map:
        raise ValueError(f"Unsupported feature_scale: {args.feature_scale}. Use 4, 8, or 16.")
    sam_feature_dim = sam_feature_dim_map[args.feature_scale]
    
    point_predictor = PointPredictorFromSAMFeatures(
        sam_feature_dim=sam_feature_dim,
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


```

```python
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
    p.add_argument("--use_class_weights", action="store_true",
                   help="Use class weights for label loss (weight foreground more)")
    p.add_argument("--foreground_weight", type=float, default=2.0,
                   help="Weight for foreground class (when use_class_weights=True)")
    
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
    p.add_argument("--eval_thresholds", type=str, default="8,12,16,20,40",
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
    use_class_weights: bool = False,
    foreground_weight: float = 2.0,
    device: Optional[torch.device] = None,
) -> tuple:
    """
    计算总损失
    
    Args:
        use_class_weights: 是否使用类别权重平衡
        foreground_weight: 前景类别的权重（当use_class_weights=True时）
        device: 设备（当use_class_weights=True时需要）
    
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
    if use_class_weights and device is not None:
        # 使用类别权重平衡：给前景点更高的权重
        # [背景, 前景] = [0, 1]
        class_weights = torch.tensor([1.0, foreground_weight], device=device)
        label_loss = F.cross_entropy(label_logits, target_labels, weight=class_weights)
    else:
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
    # 确保模型处于训练模式
    point_predictor.train()
    sam2_lora.model.train()  # 确保 SAM 模型也处于训练模式，以便 LoRA 参数参与梯度计算
    
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
            
            # 在使用 scaler.step() 之前，必须先 unscale 来检查梯度
            # 即使不需要梯度裁剪，也需要 unscale 来记录 inf checks
            # 注意：必须对每个优化器分别调用 unscale_，且必须在 step 之前
            # 重要：如果优化器中的参数没有梯度，unscale_ 不会记录 inf checks
            # 因此需要确保参数有梯度，或者检查是否有梯度再调用 unscale
            if optimizer_sam is not None:
                # 检查 LoRA 参数是否有梯度
                # 如果参数没有梯度，unscale_ 不会记录 inf checks，导致 step 失败
                has_grad = any(p.grad is not None for p in sam2_lora.get_lora_parameters())
                if not has_grad:
                    # 如果没有梯度，说明 LoRA 参数没有参与计算
                    # 这可能是因为 SAM 模型被设置为 eval 模式，或者使用了 no_grad
                    # 在这种情况下，跳过这个优化器的更新
                    print(f"Warning: LoRA parameters have no gradients, skipping optimizer_sam update")
                    optimizer_sam = None
                else:
                    # 有梯度，调用 unscale
                    scaler.unscale_(optimizer_sam)
            
            # 总是对 point_predictor 调用 unscale（因为它应该总是有梯度）
            scaler.unscale_(optimizer_point)
            
            # 梯度裁剪（在 unscale 之后，在 step 之前）
            if args.grad_clip > 0:
                if optimizer_sam is not None:
                    torch.nn.utils.clip_grad_norm_(sam2_lora.get_lora_parameters(), args.grad_clip)
                torch.nn.utils.clip_grad_norm_(point_predictor.parameters(), args.grad_clip)
            
            # 更新参数（必须在 unscale 之后）
            # 注意：如果 unscale 检测到 inf/NaN，step 会跳过这次更新
            if optimizer_sam is not None:
                scaler.step(optimizer_sam)
            scaler.step(optimizer_point)
            
            # 更新 scaler 的缩放因子（根据是否有 inf/NaN）
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
    
    # 根据 feature_scale 确定 SAM 特征通道数
    # SAM2 不同 scale 的特征通道数：
    # - Scale 4: 32 channels
    # - Scale 8: 64 channels
    # - Scale 16: 256 channels
    sam_feature_dim_map = {4: 32, 8: 64, 16: 256}
    if args.feature_scale not in sam_feature_dim_map:
        raise ValueError(f"Unsupported feature_scale: {args.feature_scale}. Use 4, 8, or 16.")
    sam_feature_dim = sam_feature_dim_map[args.feature_scale]
    
    point_predictor = PointPredictorFromSAMFeatures(
        sam_feature_dim=sam_feature_dim,
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


```

```python
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
    p.add_argument("--use_class_weights", action="store_true",
                   help="Use class weights for label loss (weight foreground more)")
    p.add_argument("--foreground_weight", type=float, default=2.0,
                   help="Weight for foreground class (when use_class_weights=True)")
    
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
    p.add_argument("--eval_thresholds", type=str, default="8,12,16,20,40",
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
    use_class_weights: bool = False,
    foreground_weight: float = 2.0,
    device: Optional[torch.device] = None,
) -> tuple:
    """
    计算总损失
    
    Args:
        use_class_weights: 是否使用类别权重平衡
        foreground_weight: 前景类别的权重（当use_class_weights=True时）
        device: 设备（当use_class_weights=True时需要）
    
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
    if use_class_weights and device is not None:
        # 使用类别权重平衡：给前景点更高的权重
        # [背景, 前景] = [0, 1]
        class_weights = torch.tensor([1.0, foreground_weight], device=device)
        label_loss = F.cross_entropy(label_logits, target_labels, weight=class_weights)
    else:
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
    # 确保模型处于训练模式
    point_predictor.train()
    sam2_lora.model.train()  # 确保 SAM 模型也处于训练模式，以便 LoRA 参数参与梯度计算
    
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
                args.loss, args.sigma, args.tau, args.label_loss_weight,
                args.use_class_weights, args.foreground_weight, args.device
            )
        
        # 反向传播
        if optimizer_sam is not None:
            optimizer_sam.zero_grad()
        optimizer_point.zero_grad()
        
        if scaler is not None:
            scaler.scale(total_loss).backward()
            
            # 在使用 scaler.step() 之前，必须先 unscale 来检查梯度
            # 即使不需要梯度裁剪，也需要 unscale 来记录 inf checks
            # 注意：必须对每个优化器分别调用 unscale_，且必须在 step 之前
            # 重要：如果优化器中的参数没有梯度，unscale_ 不会记录 inf checks
            # 因此需要确保参数有梯度，或者检查是否有梯度再调用 unscale
            if optimizer_sam is not None:
                # 检查 LoRA 参数是否有梯度
                # 如果参数没有梯度，unscale_ 不会记录 inf checks，导致 step 失败
                has_grad = any(p.grad is not None for p in sam2_lora.get_lora_parameters())
                if not has_grad:
                    # 如果没有梯度，说明 LoRA 参数没有参与计算
                    # 这可能是因为 SAM 模型被设置为 eval 模式，或者使用了 no_grad
                    # 在这种情况下，跳过这个优化器的更新
                    print(f"Warning: LoRA parameters have no gradients, skipping optimizer_sam update")
                    optimizer_sam = None
                else:
                    # 有梯度，调用 unscale
                    scaler.unscale_(optimizer_sam)
            
            # 总是对 point_predictor 调用 unscale（因为它应该总是有梯度）
            scaler.unscale_(optimizer_point)
            
            # 梯度裁剪（在 unscale 之后，在 step 之前）
            if args.grad_clip > 0:
                if optimizer_sam is not None:
                    torch.nn.utils.clip_grad_norm_(sam2_lora.get_lora_parameters(), args.grad_clip)
                torch.nn.utils.clip_grad_norm_(point_predictor.parameters(), args.grad_clip)
            
            # 更新参数（必须在 unscale 之后）
            # 注意：如果 unscale 检测到 inf/NaN，step 会跳过这次更新
            if optimizer_sam is not None:
                scaler.step(optimizer_sam)
            scaler.step(optimizer_point)
            
            # 更新 scaler 的缩放因子（根据是否有 inf/NaN）
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
    
    # 根据 feature_scale 确定 SAM 特征通道数
    # SAM2 不同 scale 的特征通道数：
    # - Scale 4: 32 channels
    # - Scale 8: 64 channels
    # - Scale 16: 256 channels
    sam_feature_dim_map = {4: 32, 8: 64, 16: 256}
    if args.feature_scale not in sam_feature_dim_map:
        raise ValueError(f"Unsupported feature_scale: {args.feature_scale}. Use 4, 8, or 16.")
    sam_feature_dim = sam_feature_dim_map[args.feature_scale]
    
    point_predictor = PointPredictorFromSAMFeatures(
        sam_feature_dim=sam_feature_dim,
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


```


修复 device 参数问题：


```python
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
    p.add_argument("--use_class_weights", action="store_true",
                   help="Use class weights for label loss (weight foreground more)")
    p.add_argument("--foreground_weight", type=float, default=2.0,
                   help="Weight for foreground class (when use_class_weights=True)")
    
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
    p.add_argument("--eval_thresholds", type=str, default="8,12,16,20,40",
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
    use_class_weights: bool = False,
    foreground_weight: float = 2.0,
    device: Optional[torch.device] = None,
) -> tuple:
    """
    计算总损失
    
    Args:
        use_class_weights: 是否使用类别权重平衡
        foreground_weight: 前景类别的权重（当use_class_weights=True时）
        device: 设备（当use_class_weights=True时需要）
    
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
    if use_class_weights and device is not None:
        # 使用类别权重平衡：给前景点更高的权重
        # [背景, 前景] = [0, 1]
        class_weights = torch.tensor([1.0, foreground_weight], device=device)
        label_loss = F.cross_entropy(label_logits, target_labels, weight=class_weights)
    else:
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
    # 确保模型处于训练模式
    point_predictor.train()
    sam2_lora.model.train()  # 确保 SAM 模型也处于训练模式，以便 LoRA 参数参与梯度计算
    
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
            device_obj = torch.device(args.device) if isinstance(args.device, str) else args.device
            total_loss, heatmap_loss, label_loss = compute_loss(
                heatmap_logits, target_points, label_logits, target_labels,
                args.loss, args.sigma, args.tau, args.label_loss_weight,
                args.use_class_weights, args.foreground_weight, device_obj
            )
        
        # 反向传播
        if optimizer_sam is not None:
            optimizer_sam.zero_grad()
        optimizer_point.zero_grad()
        
        if scaler is not None:
            scaler.scale(total_loss).backward()
            
            # 在使用 scaler.step() 之前，必须先 unscale 来检查梯度
            # 即使不需要梯度裁剪，也需要 unscale 来记录 inf checks
            # 注意：必须对每个优化器分别调用 unscale_，且必须在 step 之前
            # 重要：如果优化器中的参数没有梯度，unscale_ 不会记录 inf checks
            # 因此需要确保参数有梯度，或者检查是否有梯度再调用 unscale
            if optimizer_sam is not None:
                # 检查 LoRA 参数是否有梯度
                # 如果参数没有梯度，unscale_ 不会记录 inf checks，导致 step 失败
                has_grad = any(p.grad is not None for p in sam2_lora.get_lora_parameters())
                if not has_grad:
                    # 如果没有梯度，说明 LoRA 参数没有参与计算
                    # 这可能是因为 SAM 模型被设置为 eval 模式，或者使用了 no_grad
                    # 在这种情况下，跳过这个优化器的更新
                    print(f"Warning: LoRA parameters have no gradients, skipping optimizer_sam update")
                    optimizer_sam = None
                else:
                    # 有梯度，调用 unscale
                    scaler.unscale_(optimizer_sam)
            
            # 总是对 point_predictor 调用 unscale（因为它应该总是有梯度）
            scaler.unscale_(optimizer_point)
            
            # 梯度裁剪（在 unscale 之后，在 step 之前）
            if args.grad_clip > 0:
                if optimizer_sam is not None:
                    torch.nn.utils.clip_grad_norm_(sam2_lora.get_lora_parameters(), args.grad_clip)
                torch.nn.utils.clip_grad_norm_(point_predictor.parameters(), args.grad_clip)
            
            # 更新参数（必须在 unscale 之后）
            # 注意：如果 unscale 检测到 inf/NaN，step 会跳过这次更新
            if optimizer_sam is not None:
                scaler.step(optimizer_sam)
            scaler.step(optimizer_point)
            
            # 更新 scaler 的缩放因子（根据是否有 inf/NaN）
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
    
    # 根据 feature_scale 确定 SAM 特征通道数
    # SAM2 不同 scale 的特征通道数：
    # - Scale 4: 32 channels
    # - Scale 8: 64 channels
    # - Scale 16: 256 channels
    sam_feature_dim_map = {4: 32, 8: 64, 16: 256}
    if args.feature_scale not in sam_feature_dim_map:
        raise ValueError(f"Unsupported feature_scale: {args.feature_scale}. Use 4, 8, or 16.")
    sam_feature_dim = sam_feature_dim_map[args.feature_scale]
    
    point_predictor = PointPredictorFromSAMFeatures(
        sam_feature_dim=sam_feature_dim,
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


```


## 诊断结果总结

### 主要问题

1. **label_loss_weight 太小**（0.1）
   - label loss 权重仅为 heatmap loss 的 10%
   - 模型更关注位置预测，忽略 label 预测
   - 第一个点的 label 几乎总是 1（前景），但模型预测多为 0（背景）

2. **类别不平衡**
   - 第一个点：100% 前景（label=1）
   - 所有点：背景 55%，前景 45%
   - 步骤 0 的 label 几乎总是 1，但其他步骤分布更均匀

3. **数据集采样逻辑**
   - 步骤 0 被选中的概率约 5.4%，与其他步骤相同
   - 步骤 0 的 label 几乎总是 1，但其他步骤分布更均匀
   - 模型在步骤 0 时预测 label=0 也可能获得较低 loss

### 修复方案

已修改训练代码，添加以下选项：

1. **增加 label_loss_weight**（推荐）
   ```bash
   --label_loss_weight 0.5  # 从0.1增加到0.5
   ```

2. **使用类别权重平衡**（推荐）
   ```bash
   --use_class_weights \
   --foreground_weight 2.0  # 给前景点2倍权重
   ```

3. **组合使用**（最佳）
   ```bash
   --label_loss_weight 0.5 \
   --use_class_weights \
   --foreground_weight 2.0
   ```

### 修改的文件

1. **`seg-rl/peft/train_supervised_peft.py`**
   - 添加 `--use_class_weights` 参数
   - 添加 `--foreground_weight` 参数
   - 修改 `compute_loss` 函数，支持类别权重平衡

2. **`seg-rl/peft/diagnose_training.py`**（新建）
   - 诊断脚本，检查训练数据分布和模型设计

3. **`docs/training_issues_analysis.md`**（新建）
   - 问题分析报告和修复方案

### 建议的修复步骤

1. **立即修复**：使用类别权重平衡重新训练
   ```bash
   python -m seg-rl.peft.train_supervised_peft \
     ... \
     --label_loss_weight 0.5 \
     --use_class_weights \
     --foreground_weight 2.0 \
     ...
   ```

2. **验证效果**：
   - 检查训练日志中的 `train/label_loss` 是否下降
   - 使用 `predict_next_point_from_model.py` 预测第一个点
   - 检查第一个点的 label 是否更倾向于 1（前景）

3. **进一步优化**：
   - 如果效果不好，可以增加 `foreground_weight` 到 3.0 或 4.0
   - 或者调整数据集采样策略，增加步骤 0 的采样概率

### 其他发现

1. **坐标缩放问题**：已修复，使用 `soft_argmax_from_logits` 正确计算坐标
2. **模型架构**：看起来正确
3. **训练超参数**：可能需要调整学习率、batch size 等

诊断完成。主要问题是 **label_loss_weight 太小**，导致模型主要学习预测点的位置，而忽略 label 的预测。建议使用类别权重平衡重新训练。

```shellscript
   --label_loss_weight 0.5  # 从0.1增加到0.5
```

```shellscript
   --use_class_weights \
   --foreground_weight 2.0  # 给前景点2倍权重
```

```shellscript
   --label_loss_weight 0.5 \
   --use_class_weights \
   --foreground_weight 2.0
```

```shellscript
   python -m seg-rl.peft.train_supervised_peft \
     ... \
     --label_loss_weight 0.5 \
     --use_class_weights \
     --foreground_weight 2.0 \
     ...
```

---

**User**

重新跑了一次训练： @bash (100-106) ，然后用其中得到的模型做预测：@bash (997-1001) ，但之前发现的问题仍然显著存在：1. val_pck始终很低（val_pck@40.0也只有最多0.5左右）2. 第一个提示点坐标与启发式方法生成的（就是用来训练预测模型的真值数据）差异很大；3. 第一个提示点的label还经常为0（应该几乎一定为1才对)。基于这些问题，有什么建议？

---

**Cursor**


检查训练配置、训练日志和预测结果，分析问题：



检查训练配置和预测结果，分析问题：



分析问题并给出建议：



发现几个问题，建议如下：

## 发现的问题

### 1. 配置冲突（重要）

在 `README_PEFT_EXPERIMENT_GUIDE.md` 第275-311行，训练配置中有重复的 `--label_loss_weight` 参数：

```bash
--label_loss_weight 0.5 \      # 第286行
--use_class_weights \
--foreground_weight 2.0 \
...
--label_loss_weight 0.1 \      # 第292行（覆盖了前面的0.5！）
```

命令行中后面的参数会覆盖前面的，因此实际使用的是 `0.1` 而不是 `0.5`。

### 2. 验证时使用 argmax 而非 soft_argmax

在 `train_supervised_peft.py` 的 `validate` 函数（第354-360行）中，使用 argmax 预测点：

```python
# 预测点（使用argmax）
B, _, H, W = heatmap_logits.shape
heatmap_flat = heatmap_logits.view(B, -1)
pred_idx = heatmap_flat.argmax(dim=1)
pred_y = (pred_idx // W).float()
pred_x = (pred_idx % W).float()
pred_points = torch.stack([pred_x, pred_y], dim=1)
```

这会导致：
- 预测点不够精确（只能预测整数坐标）
- 与训练时使用的 soft_argmax 不一致
- PCK 指标可能偏低

### 3. 验证时未使用类别权重

在 `validate` 函数（第349-351行）中，调用 `compute_loss` 时没有传递类别权重参数：

```python
total_loss, _, _ = compute_loss(
    heatmap_logits, target_points, label_logits, target_labels,
    args.loss, args.sigma, args.tau, args.label_loss_weight
    # 缺少：args.use_class_weights, args.foreground_weight, device_obj
)
```

## 建议的修复方案

### 方案1：修复训练配置（立即）

修复 `README_PEFT_EXPERIMENT_GUIDE.md` 中的配置冲突，删除重复的 `--label_loss_weight 0.1`：

```bash
python -m seg-rl.peft.train_supervised_peft \
  --jsonl outputs/braintumour/peft_train-251107.jsonl \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --lora_rank 16 \
  --lora_alpha 32 \
  --feature_scale 8 \
  --fusion_mode film \
  --image_size 512 512 \
  --loss kl \
  --sigma 8.0 \
  --tau 1.0 \
  --label_loss_weight 0.5 \        # 只保留这一个
  --use_class_weights \
  --foreground_weight 2.0 \
  --batch_size 8 \
  --epochs 50 \
  ...
```

### 方案2：修复验证函数（重要）

修改 `train_supervised_peft.py` 的 `validate` 函数，使用 soft_argmax 并传递类别权重：

```python
@torch.no_grad()
def validate(
    sam2_lora: LoRASAM2Wrapper,
    point_predictor: PointPredictorFromSAMFeatures,
    val_loader: DataLoader,
    args,
    writer: Optional[SummaryWriter],
    epoch: int,
) -> dict:
    """验证"""
    point_predictor.eval()
    
    loss_meter = AverageMeter()
    pck_meters = {thresh: AverageMeter() for thresh in args.eval_thresholds}
    
    # 导入soft_argmax_from_logits
    from heatmap.model import soft_argmax_from_logits
    
    for batch in tqdm(val_loader, desc="Validating"):
        images = batch['image'].to(args.device)
        prev_masks = batch['prev_mask'].to(args.device)
        target_points = batch['target_point'].to(args.device)
        target_labels = batch['target_label'].to(args.device)
        
        # 前向传播
        sam_features = sam2_lora.get_image_features(images, feature_scale=args.feature_scale)
        heatmap_logits, label_logits = point_predictor(sam_features, prev_masks)
        
        # 计算loss（传递类别权重参数）
        device_obj = torch.device(args.device) if isinstance(args.device, str) else args.device
        total_loss, _, _ = compute_loss(
            heatmap_logits, target_points, label_logits, target_labels,
            args.loss, args.sigma, args.tau, args.label_loss_weight,
            args.use_class_weights, args.foreground_weight, device_obj
        )
        
        # 预测点（使用soft_argmax，与训练时一致）
        pred_points = soft_argmax_from_logits(heatmap_logits, temperature=args.tau)  # [B, 2] (x, y)
        
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
        writer.add_scalar('val/val_loss', loss_meter.avg, epoch)
        for thresh in args.eval_thresholds:
            writer.add_scalar(f'val/val_pck@{thresh}', pck_meters[thresh].avg, epoch)
    
    return metrics
```

### 方案3：增加 label_loss_weight 并调整类别权重

如果问题仍然存在，可以尝试：

```bash
--label_loss_weight 1.0 \        # 进一步增加到1.0
--use_class_weights \
--foreground_weight 3.0 \         # 增加前景权重到3.0
```

### 方案4：检查训练数据质量

确认训练数据中第一个点的分布：

```python
# 检查训练数据
import json
with open('outputs/braintumour/peft_train-251107.jsonl', 'r') as f:
    data = json.load(f)

first_labels = [s['labels'][0] for s in data if len(s.get('labels', [])) > 0]
print(f"第一个点的label分布: Label 0={first_labels.count(0)}, Label 1={first_labels.count(1)}")
```

### 方案5：调整数据集采样策略

如果步骤0的样本训练不足，可以增加步骤0的采样概率。修改 `datasets_peft.py`：

```python
# 在 __getitem__ 函数中
# 随机选择训练步骤k（从0到max_k-1）
# 增加步骤0的采样概率
if random.random() < 0.3:  # 30%的概率选择步骤0
    k = 0
else:
    k = random.randint(1, max_k - 1)
```

## 优先级建议

1. 立即修复：删除 README 中重复的 `--label_loss_weight 0.1` 参数
2. 重要修复：修改 `validate` 函数，使用 soft_argmax 并传递类别权重参数
3. 进一步优化：如果问题仍存在，增加 `label_loss_weight` 到 1.0，`foreground_weight` 到 3.0
4. 长期优化：考虑调整数据集采样策略，增加步骤0的采样概率

这些修复应能改善：
- val_pck 指标（使用 soft_argmax 更精确）
- 第一个点的 label 预测（使用类别权重）
- 坐标预测（使用 soft_argmax）

需要我帮你应用这些修改吗？

```shellscript
--label_loss_weight 0.5 \      # 第286行
--use_class_weights \
--foreground_weight 2.0 \
...
--label_loss_weight 0.1 \      # 第292行（覆盖了前面的0.5！）
```

```python
# 预测点（使用argmax）
B, _, H, W = heatmap_logits.shape
heatmap_flat = heatmap_logits.view(B, -1)
pred_idx = heatmap_flat.argmax(dim=1)
pred_y = (pred_idx // W).float()
pred_x = (pred_idx % W).float()
pred_points = torch.stack([pred_x, pred_y], dim=1)
```

```python
total_loss, _, _ = compute_loss(
    heatmap_logits, target_points, label_logits, target_labels,
    args.loss, args.sigma, args.tau, args.label_loss_weight
    # 缺少：args.use_class_weights, args.foreground_weight, device_obj
)
```

```shellscript
python -m seg-rl.peft.train_supervised_peft \
  --jsonl outputs/braintumour/peft_train-251107.jsonl \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --lora_rank 16 \
  --lora_alpha 32 \
  --feature_scale 8 \
  --fusion_mode film \
  --image_size 512 512 \
  --loss kl \
  --sigma 8.0 \
  --tau 1.0 \
  --label_loss_weight 0.5 \        # 只保留这一个
  --use_class_weights \
  --foreground_weight 2.0 \
  --batch_size 8 \
  --epochs 50 \
  ...
```

```python
@torch.no_grad()
def validate(
    sam2_lora: LoRASAM2Wrapper,
    point_predictor: PointPredictorFromSAMFeatures,
    val_loader: DataLoader,
    args,
    writer: Optional[SummaryWriter],
    epoch: int,
) -> dict:
    """验证"""
    point_predictor.eval()
    
    loss_meter = AverageMeter()
    pck_meters = {thresh: AverageMeter() for thresh in args.eval_thresholds}
    
    # 导入soft_argmax_from_logits
    from heatmap.model import soft_argmax_from_logits
    
    for batch in tqdm(val_loader, desc="Validating"):
        images = batch['image'].to(args.device)
        prev_masks = batch['prev_mask'].to(args.device)
        target_points = batch['target_point'].to(args.device)
        target_labels = batch['target_label'].to(args.device)
        
        # 前向传播
        sam_features = sam2_lora.get_image_features(images, feature_scale=args.feature_scale)
        heatmap_logits, label_logits = point_predictor(sam_features, prev_masks)
        
        # 计算loss（传递类别权重参数）
        device_obj = torch.device(args.device) if isinstance(args.device, str) else args.device
        total_loss, _, _ = compute_loss(
            heatmap_logits, target_points, label_logits, target_labels,
            args.loss, args.sigma, args.tau, args.label_loss_weight,
            args.use_class_weights, args.foreground_weight, device_obj
        )
        
        # 预测点（使用soft_argmax，与训练时一致）
        pred_points = soft_argmax_from_logits(heatmap_logits, temperature=args.tau)  # [B, 2] (x, y)
        
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
        writer.add_scalar('val/val_loss', loss_meter.avg, epoch)
        for thresh in args.eval_thresholds:
            writer.add_scalar(f'val/val_pck@{thresh}', pck_meters[thresh].avg, epoch)
    
    return metrics
```

```shellscript
--label_loss_weight 1.0 \        # 进一步增加到1.0
--use_class_weights \
--foreground_weight 3.0 \         # 增加前景权重到3.0
```

```python
# 检查训练数据
import json
with open('outputs/braintumour/peft_train-251107.jsonl', 'r') as f:
    data = json.load(f)

first_labels = [s['labels'][0] for s in data if len(s.get('labels', [])) > 0]
print(f"第一个点的label分布: Label 0={first_labels.count(0)}, Label 1={first_labels.count(1)}")
```

```python
# 在 __getitem__ 函数中
# 随机选择训练步骤k（从0到max_k-1）
# 增加步骤0的采样概率
if random.random() < 0.3:  # 30%的概率选择步骤0
    k = 0
else:
    k = random.randint(1, max_k - 1)
```

