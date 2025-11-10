# Conv-LoRA 快速开始指南

## 🎯 一分钟了解 Conv-LoRA

Conv-LoRA = LoRA + 卷积操作，更好地保持空间信息，适合视觉任务。

**与 Late LoRA 的关系**: 互斥，只能选其一。

## ⚡ 快速开始

### 训练

```bash
python -m seg-rl.heatmap.train \
  --jsonl data.jsonl \
  --sam_dir masks/ \
  --height 512 --width 512 \
  --use_sam_encoder \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --sam_peft_method conv_lora \
  --sam_conv_lora_rank 8 \
  --sam_conv_lora_kernel_size 3 \
  --epochs 100 \
  --out_dir outputs/conv_lora_exp
```

### 推理

```bash
python seg-rl/heatmap/predict_point_sequence_with_sam.py \
  --model_path outputs/conv_lora_exp/model_epoch_100.pt \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  ... 其他参数 ...

# 自动检测为 Conv-LoRA 模型并正确加载
```

## 🔄 模式选择

| 命令参数 | 结果模式 |
|---------|---------|
| 无 `--use_sam_encoder` | 标准模型 |
| `--use_sam_encoder` | SAM 冻结 |
| `--sam_peft_method late_lora` | Late LoRA |
| `--sam_peft_method conv_lora` | Conv-LoRA 🆕 |

## 📝 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--sam_conv_lora_rank` | 8 | LoRA 秩（4-16） |
| `--sam_conv_lora_alpha` | 16.0 | 缩放因子 |
| `--sam_conv_lora_kernel_size` | 3 | 卷积核（1,3,5） |
| `--sam_conv_lora_dropout` | 0.0 | Dropout 概率 |
| `--sam_conv_lora_blocks` | None | 块索引（逗号分隔） |

## ✅ vs ❌ 

### ✅ 正确用法

```bash
# Conv-LoRA
--sam_peft_method conv_lora --sam_conv_lora_rank 8

# Late LoRA
--sam_peft_method late_lora --sam_lora_rank 8

# SAM 冻结
--use_sam_encoder  # 不指定 peft_method
```

### ❌ 错误用法

```bash
# 不能同时使用两种方法
--sam_peft_method conv_lora --sam_lora_enabled  # ❌

# 不能混用参数
--sam_lora_enabled --sam_conv_lora_rank 8  # ❌
```

## 📊 何时使用 Conv-LoRA？

✅ **推荐使用** Conv-LoRA：
- 视觉密集任务（分割、检测）
- 需要保持空间结构
- 有充足计算资源
- 追求最佳性能

✅ **推荐使用** Late LoRA：
- 计算资源有限
- 需要最少参数
- 快速原型验证
- 通用场景

## 📚 详细文档

- **完整指南**: `README_CONV_LORA.md`
- **方法对比**: `SAM_PEFT_METHODS_COMPARISON.md`
- **知识地图**: `docs/knowledge_map_index.md`

---

就这么简单！选择 Conv-LoRA 或 Late LoRA，开始训练！🚀

