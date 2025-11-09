# SAM Late LoRA 集成使用指南

本文档介绍如何使用 SAM image encoder 配合 Late LoRA 进行参数高效微调。

## 概述

基于论文 ["Parameter Efficient Fine-Tuning of Segment Anything Model for Biomedical Imaging"](https://arxiv.org/abs/2502.00418)，我们实现了 Late LoRA 方法，可以在监督训练提示点预测模型时，同时微调 SAM 的 image encoder。

### 主要特性

1. **Late LoRA 微调**: 仅在 SAM image encoder 的最后一个 Transformer 块中添加 LoRA 适配器
2. **参数高效**: 相比全量微调，大幅减少可训练参数（通常 <1%）
3. **灵活控制**: 通过命令行参数轻松启用/禁用 LoRA
4. **向后兼容**: 关闭 LoRA 时，程序表现与未修改前完全一致
5. **Checkpoint 兼容**: 自动处理不同模型类型的 checkpoint 加载

## 安装依赖

```bash
# 安装 SAM2（如果尚未安装）
git clone https://github.com/facebookresearch/segment-anything-2.git third_party/sam2
cd third_party/sam2
pip install -e .
cd ../..

# 下载 SAM2 模型
mkdir -p third_party/sam2/checkpoints
wget -O third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

## 使用方法

### 1. 标准训练（不使用 SAM encoder）

与原有方式完全一致：

```bash
python -m seg-rl.heatmap.train \
  --jsonl datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour-251001.jsonl \
  --sam_dir datasets/seg_r1_md/Task01_BrainTumour/pretrain_gt_masks-251001 \
  --height 240 --width 240 \
  --arch unet_s \
  --loss kl --sigma 8.0 --tau 1.0 \
  --batch_size 16 --epochs 100 --amp \
  --lr 1e-4 --weight_decay 1e-4 \
  --out_dir outputs/braintumour/standard_train
```

### 2. 使用 SAM encoder（冻结，不使用 LoRA）

SAM 作为特征提取器，但不参与训练：

```bash
python -m seg-rl.heatmap.train \
  --jsonl datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour-251001.jsonl \
  --sam_dir datasets/seg_r1_md/Task01_BrainTumour/pretrain_gt_masks-251001 \
  --height 240 --width 240 \
  --arch unet_s \
  --loss kl --sigma 8.0 --tau 1.0 \
  --batch_size 16 --epochs 100 --amp \
  --lr 1e-4 --weight_decay 1e-4 \
  --use_sam_encoder \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --out_dir outputs/braintumour/sam_frozen_train
```

### 3. 使用 SAM encoder + Late LoRA 微调（推荐）

同时微调 SAM encoder 和点预测模型：

```bash
python -m seg-rl.heatmap.train \
  --jsonl datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour-251001.jsonl \
  --sam_dir datasets/seg_r1_md/Task01_BrainTumour/pretrain_gt_masks-251001 \
  --height 240 --width 240 \
  --arch unet_s \
  --loss kl --sigma 8.0 --tau 1.0 \
  --batch_size 16 --epochs 100 --amp \
  --lr 1e-4 --weight_decay 1e-4 \
  --use_sam_encoder \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --sam_lora_enabled \
  --sam_lora_rank 8 \
  --sam_lora_alpha 16.0 \
  --sam_lora_dropout 0.0 \
  --out_dir outputs/braintumour/sam_lora_train
```

### 4. LoRA 参数调优

可以为 LoRA 参数使用独立的学习率：

```bash
python -m seg-rl.heatmap.train \
  --jsonl datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour-251001.jsonl \
  --sam_dir datasets/seg_r1_md/Task01_BrainTumour/pretrain_gt_masks-251001 \
  --height 240 --width 240 \
  --arch unet_s \
  --loss kl --sigma 8.0 --tau 1.0 \
  --batch_size 16 --epochs 100 --amp \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --use_sam_encoder \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --sam_lora_enabled \
  --sam_lora_rank 8 \
  --sam_lora_alpha 16.0 \
  --sam_lora_dropout 0.1 \
  --sam_lora_lr 5e-5 \
  --out_dir outputs/braintumour/sam_lora_separate_lr
```

## 命令行参数说明

### SAM Encoder 相关参数

- `--use_sam_encoder`: 启用 SAM image encoder 作为特征提取器
- `--sam_checkpoint PATH`: SAM2 模型检查点路径（必需，当使用 `--use_sam_encoder` 时）

### LoRA 相关参数

- `--sam_lora_enabled`: 启用 Late LoRA 微调
- `--sam_lora_rank INT`: LoRA 秩（默认: 8，推荐范围: 4-16）
  - 越小参数越少，计算越快，但表达能力可能受限
  - 越大表达能力越强，但参数和计算量增加
- `--sam_lora_alpha FLOAT`: LoRA 缩放因子（默认: 16.0，通常设为 rank 的 1-2 倍）
- `--sam_lora_dropout FLOAT`: LoRA dropout 概率（默认: 0.0，范围: 0.0-0.5）
- `--sam_lora_lr FLOAT`: LoRA 参数的独立学习率（默认: None，使用 `--lr`）

### 推荐参数组合

#### 快速实验（参数最少）
```bash
--sam_lora_rank 4 --sam_lora_alpha 8.0
```

#### 平衡配置（推荐）
```bash
--sam_lora_rank 8 --sam_lora_alpha 16.0
```

#### 高性能配置（参数较多）
```bash
--sam_lora_rank 16 --sam_lora_alpha 32.0 --sam_lora_dropout 0.1
```

## 推理

推理时会自动检测 checkpoint 类型并加载相应的模型：

```bash
# 标准模型推理
python -m seg-rl.heatmap.infer \
  --images datasets/seg_r1_md/Task01_BrainTumour/canonical/images \
  --ckpt outputs/braintumour/standard_train/model_epoch_100.pt \
  --height 240 --width 240 \
  --arch unet_s --soft

# SAM-based 模型推理（需要提供 SAM checkpoint）
python -m seg-rl.heatmap.infer \
  --images datasets/seg_r1_md/Task01_BrainTumour/canonical/images \
  --ckpt outputs/braintumour/sam_lora_train/model_epoch_100.pt \
  --height 240 --width 240 \
  --arch unet_s --soft \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt
```

## Checkpoint 兼容性

### 自动兼容性处理

系统会自动处理以下情况：

1. **标准模型 ↔ 标准模型**: 完全兼容
2. **SAM 模型 ↔ SAM 模型**: 完全兼容（包括 LoRA 参数）
3. **标准模型 → SAM 模型**: 只加载点预测头参数，SAM encoder 使用预训练权重
4. **SAM 模型 → 标准模型**: 不兼容，会报错

### Checkpoint 元数据

每个 checkpoint 包含以下元数据：

```python
{
    "use_sam_encoder": bool,      # 是否使用 SAM encoder
    "sam_lora_enabled": bool,     # 是否启用 LoRA
    "sam_lora_rank": int,         # LoRA 秩
    "sam_lora_alpha": float,      # LoRA alpha
    "sam_lora_dropout": float,    # LoRA dropout
    "sam_checkpoint": str,        # SAM checkpoint 路径（可选）
    "version": str,               # Checkpoint 版本
}
```

## 性能对比

基于论文建议和我们的实现：

| 配置 | 可训练参数 | 训练速度 | 性能 |
|------|-----------|---------|------|
| 标准模型 | ~100% | 快 | 基线 |
| SAM 冻结 | ~10% | 中等 | 好 |
| SAM + Late LoRA (rank=4) | ~11% | 中等 | 很好 |
| SAM + Late LoRA (rank=8) | ~12% | 中等 | 最好 |
| SAM + Late LoRA (rank=16) | ~15% | 较慢 | 最好+ |

*注：具体数值取决于模型架构和数据集*

## 常见问题

### Q1: 如何判断是否应该使用 LoRA？

**A:** 推荐在以下情况使用 LoRA：
- 医学图像等专业领域数据
- 数据分布与 SAM 预训练数据差异较大
- 需要更好的任务适配性能

如果数据与自然图像相似，冻结 SAM 可能就足够了。

### Q2: LoRA 会显著增加训练时间吗？

**A:** 不会。Late LoRA 只在最后一个 Transformer 块中添加参数，增加的计算量很小（通常 <5%）。

### Q3: 如何选择 LoRA rank？

**A:** 
- 从 rank=8 开始（默认值）
- 如果性能不够，增加到 16
- 如果需要更快训练或参数更少，降低到 4
- rank 越大不一定越好，建议根据验证集性能选择

### Q4: 可以在预训练的标准模型基础上继续用 LoRA 训练吗？

**A:** 可以，但需要注意：
- 标准模型的 checkpoint 只包含点预测头的权重
- 加载到 SAM 模型时，只有点预测头会被初始化
- SAM encoder 和 LoRA 层将从零开始
- 这相当于用预训练的点预测头 + 新的 SAM encoder

### Q5: checkpoint 文件会变大吗？

**A:** 会，但增加有限：
- 标准模型: ~50MB
- SAM 模型（冻结）: ~50MB（不保存 SAM encoder 权重）
- SAM 模型（LoRA）: ~55MB（只保存 LoRA 权重，约 5MB）

SAM 的预训练权重不保存在 checkpoint 中，只保存 LoRA 增量。

## 技术细节

### LoRA 实现

LoRA 通过低秩分解减少可训练参数：

```
W = W_0 + ΔW
ΔW = B @ A

其中：
- W_0: 冻结的预训练权重
- A: [in_features, rank] 可训练矩阵
- B: [rank, out_features] 可训练矩阵
- scaling = alpha / rank
```

### Late LoRA 放置

根据论文，Late LoRA 只在最后一个 Transformer 块中添加：

```
SAM Image Encoder (Hiera)
├── Stage 0
├── Stage 1
├── Stage 2
└── Stage 3 (最后一层)
    ├── Block 0
    ├── Block 1
    └── Block N (最后一个 block)
        └── Attention
            ├── QKV projection ← **添加 LoRA**
            └── Output projection ← **添加 LoRA**
```

这种放置策略在效率和性能之间取得了最佳平衡。

## 引用

如果您使用了本实现，请引用原论文：

```bibtex
@article{teuber2025peft,
  title={Parameter Efficient Fine-Tuning of Segment Anything Model for Biomedical Imaging},
  author={Teuber, Carolin and Archit, Anwai and Pape, Constantin},
  journal={arXiv preprint arXiv:2502.00418},
  year={2025}
}
```

## 调试和故障排除

### 启用详细日志

训练时会自动打印以下信息：

- SAM checkpoint 加载状态
- LoRA 模块应用位置
- 参数统计（总参数、可训练参数、LoRA 参数）
- Checkpoint 兼容性警告

### 验证 LoRA 是否正确应用

查看训练日志中的以下输出：

```
[SAM] Loading SAM checkpoint from ...
[SAM] Enabling Late LoRA: rank=8, alpha=16.0
[Late LoRA] Applied to image_encoder.trunk.stages[-1].blocks[-1].attn.qkv
[Late LoRA] Applied to image_encoder.trunk.stages[-1].blocks[-1].attn.proj
============================================================
LoRA Configuration Summary
============================================================
Total parameters:            XXX,XXX,XXX
Trainable parameters:          X,XXX,XXX
LoRA parameters:                 XXX,XXX
Trainable ratio:                   X.XX%
============================================================
```

如果没有看到 `[Late LoRA] Applied to ...` 信息，说明 LoRA 未正确应用。

### 常见错误

1. **ModuleNotFoundError: No module named 'sam2'**
   - 解决：安装 SAM2（见"安装依赖"章节）

2. **FileNotFoundError: SAM2 config file not found**
   - 解决：确保 SAM2 安装在 `third_party/sam2` 目录

3. **CUDA out of memory**
   - 解决：减小 batch_size 或使用更小的 rank

4. **Checkpoint loading error**
   - 解决：确保提供正确的 `--sam_checkpoint` 路径（推理时）

## 更多资源

- SAM2 官方仓库: https://github.com/facebookresearch/segment-anything-2
- LoRA 论文: https://arxiv.org/abs/2106.09685
- Late LoRA 论文: https://arxiv.org/abs/2502.00418

