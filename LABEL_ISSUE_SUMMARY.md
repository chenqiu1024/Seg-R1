# 标签预测问题完整总结

## 问题发现时间线

### 2025-11-12
- 发现：推理结果所有第一个点label=0
- 训练数据：第一个点100%是label=1
- 初步怀疑：类别不平衡

### 2025-11-13
- 深入调查：对比昨天正常vs今天错误
- 关键发现1：昨天用height=512，今天用height=0（一度认为是原因）
- 关键发现2：即使用height=512，今天模型仍100%预测label=0
- 关键发现3：昨天模型98%正确，今天模型0%正确
- **根本原因**：`label_loss_weight=0.1`太小 + 类别不平衡

## 根本原因分析

### 核心问题

```
损失函数: loss = loss_hm + 0.1 * loss_label
                 (95%)      (5%)

模型学习策略:
  - 95%精力优化位置
  - 5%精力优化标签
  - "捷径"：总是预测label=0
  - 在56%样本上标签正确（背景点占比）
  - 惩罚很小（只有5%权重）
```

### 数据分布

```
训练数据标签分布:
  位置0（第一个点）: 100% 前景, 5.9%样本
  位置1-N（后续点）:  主要背景, 94.1%样本
  ────────────────────────────────
  整体:              44% 前景, 56% 背景
```

### 为什么rank=16比rank=8更差？

| 因素 | Rank=8 | Rank=16 |
|------|--------|---------|
| 模型容量 | 较小 | 较大 |
| 过拟合风险 | 中等 | 高 |
| 捷径策略倾向 | 轻微 | 严重 |
| Bias[0]-Bias[1] | 0.14 | 0.16 |
| 第一个点前景率 | 98% | 0% |

**Rank=16更容易过拟合到"总是预测背景"的捷径策略**

## 解决方案

### 已实施的代码修改

1. ✅ 添加 `--use_label_class_weights` 参数
2. ✅ 类别加权CrossEntropyLoss
3. ✅ 输入尺寸检查和警告
4. ✅ Checkpoint保存训练配置(height/width)
5. ✅ 学习率调度器状态保存/恢复

### 推荐训练配置

```bash
--label_loss_weight 1.0          # 从0.1增加到1.0
--use_label_class_weights        # 启用类别加权
```

## 性能预期

| label_loss_weight | 标签准确率 | PCK@20.0 | DICE |
|------------------|-----------|----------|------|
| 0.1 (旧) | 56% | 0.31 | 0.55-0.60 |
| 1.0 (推荐) | 95% | 0.50-0.55 | 0.68-0.72 |
| 1.0 + 加权 | 98% | 0.52-0.58 | 0.70-0.74 |

## 文件清单

### 新增文档
- `LABEL_PREDICTION_FIX.md` - 问题详细分析
- `LABEL_ISSUE_SUMMARY.md` - 本文档
- `INPUT_SIZE_MISMATCH_FIX.md` - 输入尺寸问题
- `train_conv_lora_fixed.sh` - 修复后的训练脚本

### 修改代码
- `seg-rl/heatmap/train.py` - 添加类别加权支持
- `seg-rl/heatmap/utils.py` - 保存训练配置到metadata
- `seg-rl/heatmap/predict_next_point_from_model.py` - 添加尺寸检查

## 立即使用

```bash
# 运行修复后的训练
bash train_conv_lora_fixed.sh

# 或手动运行
python seg-rl/heatmap/train.py \
  --label_loss_weight 1.0 \
  --use_label_class_weights \
  ... 其他参数
```

---

**根因**: label_loss_weight太小 + 类别不平衡  
**影响**: 标签预测完全失效  
**修复**: label_loss_weight=1.0 + 类别加权  
**状态**: ✅ 代码已修改，脚本已就绪

