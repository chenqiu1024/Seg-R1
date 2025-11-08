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

