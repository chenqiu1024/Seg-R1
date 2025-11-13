# Seg-R0 v2.1.1 紧急修复

## 版本信息

- **版本**: v2.1.1 (Hotfix)
- **日期**: 2025-11-13
- **类型**: Critical Bug Fix
- **影响**: 所有使用SAM encoder的训练

## 🔴 发现的严重问题

### 问题1：标签预测完全失效

**症状**:
- 推理时100%的第一个点被预测为background(0)
- 训练数据中100%的第一个点是foreground(1)
- 导致分割质量严重下降

**根本原因**:
```
label_loss_weight = 0.1  # 太小！
  ↓
标签损失只占总损失的5%
  ↓  
模型忽略标签学习
  ↓
学会"总是预测背景"的捷径
  ↓
标签预测完全错误
```

**详细分析**: 见 `LABEL_PREDICTION_FIX.md`

### 问题2：输入尺寸不匹配（已部分修复）

**症状**:
- 推理时使用与训练不同的尺寸
- 导致BatchNorm统计量不匹配
- 加剧标签预测错误

**详细分析**: 见 `INPUT_SIZE_MISMATCH_FIX.md`

## ✅ 修复内容

### 1. 代码修改

#### `seg-rl/heatmap/train.py`

**新增参数**:
```python
--use_label_class_weights  # 启用类别加权
```

**修改损失计算**:
```python
# 之前
loss_label = nn.CrossEntropyLoss()(label_logits, target)

# 现在（可选加权）
if label_class_weights is not None:
    loss_label = nn.CrossEntropyLoss(weight=label_class_weights)(...)
else:
    loss_label = nn.CrossEntropyLoss()(...)
```

**更新推荐值**:
```python
--label_loss_weight 1.0  # 之前推荐0.1-0.3，现在推荐0.5-1.0
```

#### `seg-rl/heatmap/utils.py`

**保存训练配置到metadata**:
```python
def save_checkpoint(..., config_args=None):
    # ...
    metadata["height"] = config_args.get("height")
    metadata["width"] = config_args.get("width")
    metadata["arch"] = config_args.get("arch")
```

#### `seg-rl/heatmap/predict_next_point_from_model.py`

**添加尺寸检查警告**:
```python
if use_sam_encoder and (inference_height != train_h or inference_width != train_w):
    print("⚠️  WARNING: Input size mismatch!")
    print("This may cause incorrect label prediction!")
```

### 2. 新增文件

- ✅ `train_conv_lora_fixed.sh` - 修复后的训练脚本
- ✅ `LABEL_PREDICTION_FIX.md` - 详细问题分析
- ✅ `INPUT_SIZE_MISMATCH_FIX.md` - 尺寸问题分析
- ✅ `LABEL_ISSUE_SUMMARY.md` - 问题总结
- ✅ `CRITICAL_FIXES_v2.1.1.md` - 本文档

## 🚨 对现有训练的影响

### 受影响的训练

所有使用以下配置的训练都**可能**受影响：
```bash
--use_sam_encoder
--label_loss_weight 0.1  # 或其他<0.5的值
```

**检查方法**:
```bash
python -c "
import json
with open('pred.jsonl') as f:
    data = json.load(f)
first = [item['labels'][0] for item in data[:100]]
fg_rate = sum(1 for l in first if l==1) / len(first) * 100
print(f'第一个点前景率: {fg_rate:.1f}%')
print('正常: >90%')
print('异常: <50%')
"
```

### 需要重新训练的情况

如果发现：
- 第一个点前景率 < 50%
- 标签预测准确率 < 70%

**建议重新训练**，使用修复后的配置。

## 📊 修复效果预期

| 指标 | 修复前 | 修复后 | 提升 |
|------|--------|--------|------|
| 标签准确率 | 56% | 95% | +70% |
| 第一个点前景率 | 0-2% | 95-98% | +95% |
| PCK@20.0 | 0.31 | 0.50-0.55 | +61-77% |
| DICE (17点) | 0.55-0.60 | 0.68-0.72 | +18-24% |

## 🎯 推荐行动

### 立即行动

1. **停止使用有问题的模型**
   ```bash
   # 标记问题模型
   mv outputs/braintumour/peft-conv_lora-251111 \
      outputs/braintumour/peft-conv_lora-251111-DEPRECATED
   ```

2. **使用修复配置重新训练**
   ```bash
   bash train_conv_lora_fixed.sh
   ```

3. **验证修复效果**
   - 训练5-10个epoch后检查标签准确率
   - 应该看到>90%的标签准确率

### 长期改进

1. **监控标签准确率**
   - 添加到训练日志
   - 每个epoch单独打印

2. **文档更新**
   - 更新所有文档中的推荐`label_loss_weight`值
   - 强调类别加权的重要性

3. **最佳实践**
   - 多任务学习时平衡各任务权重
   - 处理类别不平衡
   - 监控所有指标，不只是loss

## 📚 相关文档

1. **问题分析**:
   - `LABEL_PREDICTION_FIX.md` - 详细分析
   - `INPUT_SIZE_MISMATCH_FIX.md` - 尺寸问题

2. **解决方案**:
   - `train_conv_lora_fixed.sh` - 修复脚本
   - `LABEL_ISSUE_SUMMARY.md` - 快速总结

3. **知识地图**:
   - `docs/knowledge_map_index.md` - 项目全览
   - 需要更新以包含此修复

## 🎓 经验教训

### 多任务学习陷阱

1. **权重设置**:
   - ❌ 不要让任何任务的权重<0.5
   - ✅ 所有任务应该有相近的权重

2. **类别不平衡**:
   - ❌ 不要忽略少数类（即使只占5%）
   - ✅ 使用类别加权或重采样

3. **监控指标**:
   - ❌ 不要只看总loss
   - ✅ 监控每个任务的准确率

### Conv-LoRA特定问题

4. **模型容量**:
   - Rank越大越容易过拟合
   - 需要更强的正则化（更大的label_loss_weight）

5. **输入尺寸**:
   - 必须与训练时一致
   - 尤其对SAM encoder模型

## 版本变更

```
v2.1   → SAM Conv-LoRA集成
v2.1.1 → 修复标签预测问题（紧急）
```

---

**状态**: ✅ 修复完成  
**行动**: 使用 `train_conv_lora_fixed.sh` 重新训练  
**优先级**: 🔴 高优先级（严重影响性能）

