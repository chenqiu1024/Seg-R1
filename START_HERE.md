# 🚀 从这里开始

## 欢迎使用 Seg-R0 项目！

### 📖 快速导航

#### 🎯 我是 AI Agent，刚接手这个项目
→ **立即阅读**: [`docs/knowledge_map_index.md`](docs/knowledge_map_index.md) ⭐⭐⭐

这个文档包含：
- 项目完整概述和架构
- 所有模块和文件索引
- 常见任务和命令模板
- 故障排除指南
- v2.0 最新功能说明

#### 🔬 我想使用 SAM PEFT 方法训练模型
→ **首先阅读**: [`seg-rl/heatmap/SAM_PEFT_METHODS_COMPARISON.md`](seg-rl/heatmap/SAM_PEFT_METHODS_COMPARISON.md) ⭐⭐⭐

帮助您选择：
- Late LoRA（参数少，速度快）
- Conv-LoRA（性能好，视觉任务推荐）🆕

→ **Late LoRA 详细文档**: [`seg-rl/heatmap/README_SAM_LORA.md`](seg-rl/heatmap/README_SAM_LORA.md)

→ **Conv-LoRA 详细文档**: [`seg-rl/heatmap/README_CONV_LORA.md`](seg-rl/heatmap/README_CONV_LORA.md) 🆕

#### 📝 我想查看实验配置和参数
→ **使用**:
```bash
python seg-rl/heatmap/show_experiment_config.py <config.json>
```

→ **阅读**: [`seg-rl/heatmap/QUICK_START_PARAM_TRACKING.md`](seg-rl/heatmap/QUICK_START_PARAM_TRACKING.md)

#### 🏃 我想快速开始训练
→ **运行示例**:
```bash
bash seg-rl/heatmap/example_train_with_lora.sh
```

→ **或查看**: [`seg-rl/README_SEG_RL_INSTRUCTIONS.md`](seg-rl/README_SEG_RL_INSTRUCTIONS.md)

#### 🔧 我遇到了问题
→ **查看**: [`docs/knowledge_map_index.md`](docs/knowledge_map_index.md) 的"故障排除"章节

→ **或搜索**: `docs/cursor_*.md` 历史对话记录

---

## ⚡ 最小示例

### 训练（标准模式）
```bash
python -m seg-rl.heatmap.train \
  --jsonl datasets/my_data/train.jsonl \
  --sam_dir datasets/my_data/sam_masks \
  --height 512 --width 512 \
  --arch unet_s \
  --loss kl --sigma 8.0 --tau 1.0 \
  --epochs 100 --batch_size 16 --amp \
  --out_dir outputs/my_exp
```

### 训练（SAM Late LoRA 模式）v2.0
```bash
python -m seg-rl.heatmap.train \
  --jsonl datasets/my_data/train.jsonl \
  --sam_dir datasets/my_data/sam_masks \
  --height 512 --width 512 \
  --epochs 100 --batch_size 16 --amp \
  --use_sam_encoder \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --sam_peft_method late_lora \
  --sam_lora_rank 8 --sam_lora_alpha 16.0 \
  --out_dir outputs/my_exp_late_lora
```

### 训练（SAM Conv-LoRA 模式）v2.1 🆕
```bash
python -m seg-rl.heatmap.train \
  --jsonl datasets/my_data/train.jsonl \
  --sam_dir datasets/my_data/sam_masks \
  --height 512 --width 512 \
  --epochs 100 --batch_size 16 --amp \
  --use_sam_encoder \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --sam_peft_method conv_lora \
  --sam_conv_lora_rank 8 --sam_conv_lora_alpha 16.0 \
  --sam_conv_lora_kernel_size 3 \
  --out_dir outputs/my_exp_conv_lora
```

### 推理
```bash
python seg-rl/heatmap/predict_point_sequence_with_sam.py \
  --model_path outputs/my_exp/model_epoch_100.pt \
  --images_dir datasets/test/images \
  --masks_dir datasets/test/masks \
  --output_jsonl outputs/pred/results.jsonl \
  --sam_masks_dir outputs/pred \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --num_points 17
```

---

## ⚠️ 重要提示

### 必须记住的事项

1. **tau 参数**: 绝对不要使用 `--tau 0.0`（会导致 loss=nan）
   - ✅ 使用 `--tau 1.0`（默认推荐）

2. **SAM checkpoint**: 使用 SAM encoder 时必须提供
   - 训练时: `--sam_checkpoint <path>`
   - 推理时: `--sam_checkpoint <path>`

3. **参数自动保存**: v2.0 自动记录所有实验配置
   - 训练: `<out_dir>/training_args.json`
   - 推理: `<sam_masks_dir>/inference_args.json`

4. **模型兼容性**: 推理时会自动检测模型类型（v2.0）
   - SAM 模型需要 SAM checkpoint
   - 标准模型不需要

---

## 📚 完整文档列表

### 核心文档（必读）

1. [`docs/knowledge_map_index.md`](docs/knowledge_map_index.md) - **知识地图索引**（本项目导航中心）⭐⭐⭐
2. [`seg-rl/heatmap/README_SAM_LORA.md`](seg-rl/heatmap/README_SAM_LORA.md) - SAM Late LoRA 使用指南 ⭐⭐⭐
3. [`seg-rl/heatmap/QUICK_START_PARAM_TRACKING.md`](seg-rl/heatmap/QUICK_START_PARAM_TRACKING.md) - 参数追踪快速指南 ⭐⭐

### 功能文档

4. [`LATE_LORA_INTEGRATION_SUMMARY.md`](LATE_LORA_INTEGRATION_SUMMARY.md) - SAM LoRA 集成总结
5. [`PARAM_TRACKING_FEATURE.md`](PARAM_TRACKING_FEATURE.md) - 参数追踪功能概览
6. [`seg-rl/heatmap/EXPERIMENT_TRACKING.md`](seg-rl/heatmap/EXPERIMENT_TRACKING.md) - 实验追踪完整文档

### 技术文档

7. [`seg-rl/heatmap/CHANGELOG_SAM_LORA.md`](seg-rl/heatmap/CHANGELOG_SAM_LORA.md) - SAM LoRA 修改日志
8. [`seg-rl/heatmap/FIX_INFERENCE_SAM_LORA.md`](seg-rl/heatmap/FIX_INFERENCE_SAM_LORA.md) - 推理修复说明
9. [`seg-rl/heatmap/PARAM_SAVING_SUMMARY.md`](seg-rl/heatmap/PARAM_SAVING_SUMMARY.md) - 参数保存实现细节

---

## 版本

- **当前版本**: v2.0
- **最后更新**: 2025-11-09
- **主要特性**: SAM Late LoRA + 实验参数追踪

---

**开始探索项目，祝您使用愉快！** 🎉

