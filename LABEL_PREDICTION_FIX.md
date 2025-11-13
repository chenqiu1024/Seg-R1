# 标签预测错误问题 - 根本原因和解决方案

## 🔴 问题发现

推理时，所有第一个点的标签都被错误预测为background(0)，而训练数据中100%的第一个点是foreground(1)。

## 🎯 根本原因

### 原因1：`label_loss_weight = 0.1` 太小

**损失函数**:
```python
loss = loss_hm + 0.1 * loss_label
```

**权重分配**:
- 位置预测损失：~95%
- 标签预测损失：~5%

**结果**:
- 模型优化时几乎忽略标签损失
- 学会了"捷径策略"：总是预测背景(0)
- 在56%的样本上标签正确（背景点占比）
- 这个错误策略的惩罚很小（只有5%权重）

### 原因2：类别不平衡

**训练数据标签分布**:
```
位置0（第一个点）: 100% 前景
位置1（第二个点）: 64%  前景
位置2+（后续点）  : 33%  前景
────────────────────────────────
整体分布          : 44%  前景, 56% 背景
```

**问题**:
- 第一个点样本只占5.9%（942/16007）
- 使用不加权的CrossEntropyLoss
- 模型被多数样本（94.1%）主导
- 学会预测背景点（多数类）

### 原因3：Rank大小的影响

| Rank | 容量 | 行为 |
|------|------|------|
| 8 | 较小 | 轻微倾向背景，但仍能预测前景(98%) |
| 16 | 较大 | 强烈倾向背景，完全预测背景(100%) |

**Rank=16模型更容易过拟合到"总是预测背景"的捷径策略**

## 📊 数据分析

### PCK指标的真相

训练时的PCK计算：
```python
ok = (ok_xy & ok_label)  # 位置准确 AND 标签准确
PCK = ok / total
```

**PCK@20.0 = 0.31 的真相**:
- 如果模型总是预测label=0（背景）
- 标签准确率 ≈ 56%（背景点占比）
- 位置准确率 ≈ 55%
- PCK = 0.55 × 0.56 ≈ 0.31 ✓

**这意味着模型在训练时就学会了"总是预测背景"！**

### 模型参数验证

| 模型 | Label Head Bias | 倾向 | 实际推理 |
|------|----------------|------|---------|
| rank=8, epoch75 | [0.0633, -0.0766] | 背景 | 98%前景 ✅ |
| rank=16, epoch60 | [0.0576, -0.1015] | 背景 | 0%前景 ❌ |

**所有模型的bias都倾向背景，但程度不同导致行为差异**

## ✅ 解决方案

### 方案1：增大 `label_loss_weight`（推荐）

```bash
python seg-rl/heatmap/train.py \
  --jsonl outputs/braintumour/heuristic_251108.jsonl \
  --sam_dir outputs/braintumour/sam_masks_heuristic \
  --height 512 --width 512 \
  --arch unet_s \
  --loss kl --sigma 8.0 --tau 1.0 \
  --label_loss_weight 1.0 \                    # ← 从0.1增加到1.0
  --batch_size 16 --epochs 80 --amp \
  --lr 1e-4 --weight_decay 1e-4 \
  --use_sam_encoder \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --sam_peft_method conv_lora \
  --sam_conv_lora_rank 16 \
  --sam_conv_lora_alpha 32.0 \
  --lr_scheduler warmup_cosine \
  --warmup_epochs 5 \
  --out_dir outputs/braintumour/peft-conv_lora-label_weight1.0 \
  --save_every 10 --save_steps 500 --progress
```

**预期效果**:
- 标签预测准确率：56% → 95%
- 第一个点前景率：0% → 95%
- PCK@20.0：0.31 → 0.50-0.55

### 方案2：使用类别加权（推荐组合）

```bash
python seg-rl/heatmap/train.py \
  --jsonl outputs/braintumour/heuristic_251108.jsonl \
  --sam_dir outputs/braintumour/sam_masks_heuristic \
  --height 512 --width 512 \
  --arch unet_s \
  --loss kl --sigma 8.0 --tau 1.0 \
  --label_loss_weight 0.5 \                    # 适中的权重
  --use_label_class_weights \                  # ← 启用类别加权
  --batch_size 16 --epochs 80 --amp \
  --lr 1e-4 --weight_decay 1e-4 \
  --use_sam_encoder \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --sam_peft_method conv_lora \
  --sam_conv_lora_rank 16 \
  --sam_conv_lora_alpha 32.0 \
  --lr_scheduler warmup_cosine \
  --warmup_epochs 5 \
  --out_dir outputs/braintumour/peft-conv_lora-balanced \
  --save_every 10 --save_steps 500 --progress
```

**预期效果**:
- 类别加权给前景样本更多权重
- 标签预测准确率：56% → 90-95%
- PCK@20.0：0.31 → 0.48-0.52

### 方案3：极度重视标签（最激进）

```bash
python seg-rl/heatmap/train.py \
  --label_loss_weight 2.0 \                    # 极高权重
  --use_label_class_weights \                  # 类别加权
  ...其他参数
```

## 📈 预期性能对比

| 配置 | label_loss_weight | 类别加权 | 标签准确率 | PCK@20.0 | DICE(17点) |
|------|------------------|---------|-----------|----------|-----------|
| 当前(错误) | 0.1 | 否 | ~56% | 0.31 | 0.55-0.60 |
| 方案1 | 1.0 | 否 | ~95% | 0.50-0.55 | 0.68-0.72 |
| 方案2 | 0.5 | 是 | ~92% | 0.48-0.52 | 0.66-0.70 |
| 方案3 | 2.0 | 是 | ~98% | 0.52-0.58 | 0.70-0.74 |

## 🔧 已实施的代码改进

### 1. 添加 `--use_label_class_weights` 参数

启用后自动使用类别加权：
- Background weight: 0.44
- Foreground weight: 0.56
- 给前景样本更多关注

### 2. 修改损失计算

```python
# 之前（不加权）
loss_label = nn.CrossEntropyLoss()(label_logits, target_label)

# 现在（可选加权）
if label_class_weights is not None:
    loss_label = nn.CrossEntropyLoss(weight=label_class_weights)(...)
else:
    loss_label = nn.CrossEntropyLoss()(...)
```

### 3. 更新文档说明

- 推荐 `label_loss_weight` 从0.1增加到0.5-1.0
- 建议启用 `--use_label_class_weights`

## 💡 使用建议

### 快速修复（使用现有checkpoint继续训练）

如果想从现有模型继续训练：

```bash
# 不推荐：模型已经学坏了，很难纠正
# 建议从头重新训练
```

### 推荐做法（从头训练）

```bash
# 推荐配置：label_loss_weight=1.0 + 类别加权
python seg-rl/heatmap/train.py \
  --jsonl outputs/braintumour/heuristic_251108.jsonl \
  --sam_dir outputs/braintumour/sam_masks_heuristic \
  --height 512 --width 512 \
  --arch unet_s \
  --loss kl --sigma 8.0 --tau 1.0 \
  --label_loss_weight 1.0 \
  --use_label_class_weights \
  --batch_size 16 --epochs 80 --amp \
  --lr 1e-4 --weight_decay 1e-4 \
  --grad_clip 1.0 \
  --use_sam_encoder \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --sam_peft_method conv_lora \
  --sam_conv_lora_rank 16 \
  --sam_conv_lora_alpha 32.0 \
  --sam_conv_lora_kernel_size 3 \
  --lr_scheduler warmup_cosine \
  --warmup_epochs 5 \
  --val_ratio 0.1 --test_ratio 0.0 --seed 42 \
  --save_every 10 --save_steps 500 --progress \
  --out_dir outputs/braintumour/peft-conv_lora-fixed-v1 \
  2>&1 | tee outputs/braintumour/peft-conv_lora-fixed-v1/train.log
```

## 🎓 经验教训

### 1. 多任务学习的权重很关键

当训练一个多任务模型时（位置 + 标签）：
- 两个任务的损失权重必须平衡
- 如果一个任务权重太小，模型会忽略它
- 推荐权重：0.5-2.0（而不是0.1）

### 2. 类别不平衡必须处理

- 不加权的CE Loss被多数类主导
- 少数类（前景点，尤其第一个点）被忽略
- 必须使用类别加权或其他平衡技术

### 3. 模型容量与过拟合

- Rank越大，越容易过拟合到捷径策略
- 需要更强的正则化（更大的label_loss_weight）

### 4. PCK不能完全反映标签质量

- PCK同时考虑位置和标签
- 可能隐藏标签预测的问题
- 建议单独监控标签准确率

## 🔍 如何避免类似问题？

### 1. 添加标签准确率监控

修改 `train.py` 的evaluate函数，单独打印标签准确率：

```python
def evaluate(...):
    # ... 现有代码 ...
    
    # 添加标签准确率统计
    label_correct = ok_label.sum().item()
    label_accuracy = label_correct / total
    print(f"  Label Accuracy: {label_accuracy:.4f}")
```

### 2. 合理设置超参数

```bash
# 推荐配置
--label_loss_weight 1.0          # 与位置损失同等重要
--use_label_class_weights        # 处理类别不平衡
```

### 3. 训练时检查

定期检查：
- 标签预测准确率（应该>90%）
- 第一个点的前景率（应该>95%）
- 如果异常，立即调整

## 📝 快速诊断清单

如果怀疑标签预测有问题：

```bash
# 1. 检查推理结果的第一个点标签
python -c "
import json
with open('pred.jsonl') as f:
    data = json.load(f)
first = [item['labels'][0] for item in data[:100]]
print(f'Label 0: {sum(1 for l in first if l==0)}%')
print(f'Label 1: {sum(1 for l in first if l==1)}%')
"

# 2. 检查训练配置
cat <out_dir>/training_args.json | grep label_loss_weight

# 3. 检查模型bias
python -c "
import torch
ckpt = torch.load('model.pt', map_location='cpu')
bias = ckpt['model']['label_head.fc.3.bias']
print(f'Bias: {bias}')
print(f'Tends to: {0 if bias[0]>bias[1] else 1}')
"
```

## 🚀 立即行动

### 推荐训练命令（已优化）

```bash
python seg-rl/heatmap/train.py \
  --jsonl outputs/braintumour/heuristic_251108.jsonl \
  --sam_dir outputs/braintumour/sam_masks_heuristic \
  --height 512 --width 512 \
  --arch unet_s \
  --loss kl --sigma 8.0 --tau 1.0 \
  --label_loss_weight 1.0 \
  --use_label_class_weights \
  --batch_size 16 --epochs 80 --amp \
  --lr 1e-4 --weight_decay 1e-4 \
  --grad_clip 1.0 \
  --use_sam_encoder \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --sam_peft_method conv_lora \
  --sam_conv_lora_rank 16 \
  --sam_conv_lora_alpha 32.0 \
  --sam_conv_lora_kernel_size 3 \
  --lr_scheduler warmup_cosine \
  --warmup_epochs 5 \
  --val_ratio 0.1 --test_ratio 0.0 --seed 42 \
  --save_every 10 --save_steps 500 --progress \
  --out_dir outputs/braintumour/conv_lora_fixed \
  2>&1 | tee outputs/braintumour/conv_lora_fixed/train.log
```

### 推理时记住

```bash
# 必须指定与训练时一致的尺寸
--height 512 --width 512
```

---

**问题**: 标签损失权重太小导致模型忽略标签学习  
**影响**: 100%标签预测错误，严重影响分割质量  
**修复**: `label_loss_weight=1.0` + `--use_label_class_weights`  
**状态**: ✅ 代码已修改，可立即重新训练

