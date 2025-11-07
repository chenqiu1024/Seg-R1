#!/usr/bin/env python3
"""
PEFT点预测数据加载器

从JSONL文件加载训练数据，支持：
- 随机选择训练步骤k（0到len(points)-1）
- k=0时返回全零掩模
- k>0时从sam_masks_dir加载前一步掩模
- 数据增强（水平翻转、颜色抖动）

JSONL格式:
    {
        "image": "/path/to/image.jpg",
        "gt_mask": "/path/to/mask.png",
        "points": [[x0,y0], [x1,y1], ...],
        "labels": [1, 1, ...],
        "sam_masks_dir": "/path/to/sam_masks"
    }

SAM掩模存储结构:
    sam_masks_dir/
    ├── image_stem/
    │   ├── 0.png  # 第1个点生成的掩模
    │   ├── 1.png  # 前2个点生成的掩模
    │   └── ...

调用示例:
    from seg-rl.peft import PEFTPointDataset
    
    dataset = PEFTPointDataset(
        jsonl_path="datasets/braintumour/train.jsonl",
        image_size=(512, 512),
        augmentation=True
    )
    
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    for batch in dataloader:
        images = batch['image']          # [B, 3, H, W]
        prev_masks = batch['prev_mask']  # [B, 1, H, W]
        target_points = batch['target_point']  # [B, 2]
        target_labels = batch['target_label']  # [B]
        # ...
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class PEFTPointDataset(Dataset):
    """
    PEFT点预测训练数据集
    
    Args:
        jsonl_path: JSONL文件路径
        image_size: 图像尺寸 (H, W)
        augmentation: 是否使用数据增强
        max_points_per_sample: 每个样本最多使用多少个点（None表示全部）
        hflip_prob: 水平翻转概率
    """
    
    def __init__(
        self,
        jsonl_path: str,
        image_size: Tuple[int, int] = (512, 512),
        augmentation: bool = True,
        max_points_per_sample: Optional[int] = None,
        hflip_prob: float = 0.5,
    ):
        self.jsonl_path = jsonl_path
        self.image_size = image_size
        self.augmentation = augmentation
        self.max_points_per_sample = max_points_per_sample
        self.hflip_prob = hflip_prob
        
        # 加载JSONL
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content.startswith('['):
                # JSON数组格式
                self.samples = json.loads(content)
            else:
                # JSONL格式（逐行）
                for line in content.splitlines():
                    if line.strip():
                        self.samples.append(json.loads(line))
        
        print(f"Loaded {len(self.samples)} samples from {jsonl_path}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        image_path = sample['image']
        gt_mask_path = sample.get('gt_mask', None)
        points_seq = sample['points']
        labels_seq = sample['labels']
        sam_masks_dir = sample.get('sam_masks_dir', None)
        
        # 限制点数（如果指定）
        if self.max_points_per_sample is not None:
            max_k = min(len(points_seq), self.max_points_per_sample)
        else:
            max_k = len(points_seq)
        
        # 随机选择训练步骤k（从0到max_k-1）
        k = random.randint(0, max_k - 1)
        target_point = points_seq[k]
        target_label = labels_seq[k]
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        orig_w, orig_h = image.size
        
        # 加载前一步的掩模
        if k == 0:
            # 第一个点：前一步掩模是全零
            prev_mask = Image.new('L', (orig_w, orig_h), 0)
        else:
            # 从sam_masks_dir加载
            if sam_masks_dir is None:
                raise ValueError(f"sam_masks_dir is required for k>0, but not found in sample {idx}")
            
            stem = Path(image_path).stem
            prev_mask_path = os.path.join(sam_masks_dir, stem, f"{k-1}.png")
            
            if not os.path.exists(prev_mask_path):
                # 如果文件不存在，使用全零（防止数据不完整）
                print(f"Warning: {prev_mask_path} not found, using zero mask")
                prev_mask = Image.new('L', (orig_w, orig_h), 0)
            else:
                prev_mask = Image.open(prev_mask_path).convert('L')
        
        # 数据增强
        if self.augmentation and random.random() < self.hflip_prob:
            image = TF.hflip(image)
            prev_mask = TF.hflip(prev_mask)
            # 翻转点坐标
            target_point = [orig_w - 1 - target_point[0], target_point[1]]
        
        # Resize到目标尺寸
        image = image.resize(self.image_size, Image.BILINEAR)
        prev_mask = prev_mask.resize(self.image_size, Image.NEAREST)
        
        # 调整点坐标到resize后的尺寸
        scale_x = self.image_size[1] / orig_w
        scale_y = self.image_size[0] / orig_h
        target_point_resized = np.array([
            target_point[0] * scale_x,
            target_point[1] * scale_y
        ], dtype=np.float32)
        
        # 转换为tensor
        image_tensor = TF.to_tensor(image)  # [3, H, W], [0, 1]
        prev_mask_tensor = TF.to_tensor(prev_mask)  # [1, H, W], [0, 1]
        
        # 二值化掩模
        prev_mask_tensor = (prev_mask_tensor > 0.5).float()
        
        return {
            'image': image_tensor,
            'prev_mask': prev_mask_tensor,
            'target_point': target_point_resized,
            'target_label': target_label,
            'step': k,
            'image_path': image_path,
        }


def collate_fn_peft(batch: list) -> Dict:
    """
    自定义collate函数，处理batch
    
    Args:
        batch: list of dict from __getitem__
        
    Returns:
        batch dict with stacked tensors
    """
    images = torch.stack([item['image'] for item in batch])
    prev_masks = torch.stack([item['prev_mask'] for item in batch])
    target_points = torch.from_numpy(np.stack([item['target_point'] for item in batch]))
    target_labels = torch.tensor([item['target_label'] for item in batch], dtype=torch.long)
    steps = torch.tensor([item['step'] for item in batch], dtype=torch.long)
    image_paths = [item['image_path'] for item in batch]
    
    return {
        'image': images,
        'prev_mask': prev_masks,
        'target_point': target_points,
        'target_label': target_labels,
        'step': steps,
        'image_path': image_paths,
    }


class PEFTPointDatasetForEval(Dataset):
    """
    评估数据集：不做随机选择，而是完整遍历每个样本的所有步骤
    
    Args:
        jsonl_path: JSONL文件路径
        image_size: 图像尺寸
    """
    
    def __init__(
        self,
        jsonl_path: str,
        image_size: Tuple[int, int] = (512, 512),
    ):
        self.jsonl_path = jsonl_path
        self.image_size = image_size
        
        # 加载JSONL
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content.startswith('['):
                self.samples = json.loads(content)
            else:
                for line in content.splitlines():
                    if line.strip():
                        self.samples.append(json.loads(line))
        
        # 展开为 (sample_idx, step_k) 对
        self.items = []
        for sample_idx, sample in enumerate(self.samples):
            num_points = len(sample['points'])
            for k in range(num_points):
                self.items.append((sample_idx, k))
        
        print(f"Loaded {len(self.samples)} samples, {len(self.items)} total steps")
    
    def __len__(self) -> int:
        return len(self.items)
    
    def __getitem__(self, idx: int) -> Dict:
        sample_idx, k = self.items[idx]
        sample = self.samples[sample_idx]
        
        image_path = sample['image']
        points_seq = sample['points']
        labels_seq = sample['labels']
        sam_masks_dir = sample.get('sam_masks_dir', None)
        
        target_point = points_seq[k]
        target_label = labels_seq[k]
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        orig_w, orig_h = image.size
        
        # 加载前一步的掩模
        if k == 0:
            prev_mask = Image.new('L', (orig_w, orig_h), 0)
        else:
            stem = Path(image_path).stem
            prev_mask_path = os.path.join(sam_masks_dir, stem, f"{k-1}.png")
            if not os.path.exists(prev_mask_path):
                prev_mask = Image.new('L', (orig_w, orig_h), 0)
            else:
                prev_mask = Image.open(prev_mask_path).convert('L')
        
        # Resize
        image = image.resize(self.image_size, Image.BILINEAR)
        prev_mask = prev_mask.resize(self.image_size, Image.NEAREST)
        
        # 调整点坐标
        scale_x = self.image_size[1] / orig_w
        scale_y = self.image_size[0] / orig_h
        target_point_resized = np.array([
            target_point[0] * scale_x,
            target_point[1] * scale_y
        ], dtype=np.float32)
        
        # 转换为tensor
        image_tensor = TF.to_tensor(image)
        prev_mask_tensor = TF.to_tensor(prev_mask)
        prev_mask_tensor = (prev_mask_tensor > 0.5).float()
        
        return {
            'image': image_tensor,
            'prev_mask': prev_mask_tensor,
            'target_point': target_point_resized,
            'target_label': target_label,
            'step': k,
            'sample_idx': sample_idx,
            'image_path': image_path,
        }

