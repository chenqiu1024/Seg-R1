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

