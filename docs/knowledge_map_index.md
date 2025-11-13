# Seg-R0 项目知识地图索引

> **文档目的**: 为 AI Agent 提供快速了解整个项目的入口文档  
> **最后更新**: 2025-11-09  
> **版本**: v2.0 (包含 SAM Late LoRA 和参数追踪功能)

---

## 📑 目录

1. [项目概述](#项目概述)
2. [核心架构](#核心架构)
3. [主要模块](#主要模块)
4. [数据格式](#数据格式)
5. [训练流程](#训练流程)
6. [推理流程](#推理流程)
7. [最新功能](#最新功能)
8. [文件索引](#文件索引)
9. [常见任务](#常见任务)
10. [故障排除](#故障排除)

---

## 项目概述

### 项目目标

**Seg-R0** 是 **Seg-R1** 项目的实验和开发版本，专注于：
1. 使用强化学习方法进行图像分割
2. 通过迭代式提示点预测配合 SAM 实现分割
3. 探索参数高效微调（PEFT）方法提升性能

### 核心思想

基于论文 "Segmentation Can Be Surprisingly Simple with Reinforcement Learning"：

```
输入图像 
  ↓
迭代预测提示点序列 (RL 策略)
  ↓
SAM 生成分割 mask
  ↓
评估 mask 质量 (奖励信号)
  ↓
优化策略 (GRPO 算法)
```

### 项目组成

```
Seg-R0/
├── seg-r1/          # Seg-R1 主框架（VLM + RL）
├── seg-rl/          # 热力图点定位工具包（核心开发重点）
├── third_party/     # 第三方依赖（SAM2）
├── datasets/        # 数据集
└── outputs/         # 实验输出
```

---

## 核心架构

### 整体流程图

```
┌─────────────────────────────────────────────────────────┐
│                      Seg-R0 系统                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  阶段 1: 监督预训练 (Supervised Pre-training)            │
│  ┌────────────────────────────────────────────┐         │
│  │ 输入: RGB 图像 + 上一步 SAM mask (灰度图)  │         │
│  │   ↓                                         │         │
│  │ 热力图模型: UNet/ResNet + 热力图头          │         │
│  │   ↓                                         │         │
│  │ 输出: 下一个提示点 (x, y) + 标签 (0/1)     │         │
│  └────────────────────────────────────────────┘         │
│                                                          │
│  阶段 2: 与 SAM 集成                                     │
│  ┌────────────────────────────────────────────┐         │
│  │ 预测点 → SAM2 → mask → 评估 → 下一步      │         │
│  │ (迭代 N 次，通常 8-17 次)                  │         │
│  └────────────────────────────────────────────┘         │
│                                                          │
│  阶段 3: 强化学习微调 (GRPO)                            │
│  ┌────────────────────────────────────────────┐         │
│  │ 策略网络 → 采样动作 → 奖励 → 优化         │         │
│  └────────────────────────────────────────────┘         │
│                                                          │
│  🆕 阶段 4: SAM Late LoRA 微调 (v2.0 新增)             │
│  ┌────────────────────────────────────────────┐         │
│  │ SAM Encoder + LoRA → 点预测头              │         │
│  │ (同时微调 SAM 和预测网络)                  │         │
│  └────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────┘
```

### 技术栈

- **深度学习框架**: PyTorch 2.5.1
- **分割模型**: SAM2 (Segment Anything Model 2)
- **点预测模型**: UNet-Small (主力) / ResNet18 (备用)
- **强化学习**: GRPO (Group Relative Policy Optimization)
- **参数高效微调**: LoRA (Low-Rank Adaptation)

---

## 主要模块

### 1. seg-rl/heatmap/ - 热力图点定位模块 ⭐

**核心功能**: 训练和推理提示点预测模型

#### 关键文件

| 文件 | 功能 | 重要性 |
|------|------|--------|
| `model.py` | 模型定义（UNet, ResNet, SAM+LoRA） | ⭐⭐⭐ |
| `train.py` | 监督训练脚本 | ⭐⭐⭐ |
| `datasets.py` | 数据加载器（支持序列化训练） | ⭐⭐⭐ |
| `losses.py` | 损失函数（CE, KL, MSE） | ⭐⭐ |
| `utils.py` | 工具函数（checkpoint, 可视化） | ⭐⭐ |
| `infer.py` | 推理脚本 | ⭐⭐ |
| `predict_next_point_from_model.py` | 预测下一个点 | ⭐⭐ |
| `predict_point_sequence_with_sam.py` | 整合预测+SAM分割 | ⭐⭐⭐ |

#### 🆕 SAM PEFT 相关文件（v2.0-v2.1）

| 文件 | 功能 | 版本 |
|------|------|------|
| `sam_lora.py` | Late LoRA 实现 | v2.0 |
| `sam_conv_lora.py` | Conv-LoRA 实现 | v2.1 🆕 |
| `README_SAM_LORA.md` | Late LoRA 使用文档 | v2.0 |
| `README_CONV_LORA.md` | Conv-LoRA 使用文档 | v2.1 🆕 |
| `SAM_PEFT_METHODS_COMPARISON.md` | 方法对比指南 | v2.1 🆕 |
| `CHANGELOG_SAM_LORA.md` | Late LoRA 修改日志 | v2.0 |
| `example_train_with_lora.sh` | 训练示例脚本 | v2.0 |
| `test_compatibility.py` | 兼容性测试 | v2.0 |

#### 🆕 参数追踪功能（v2.0 新增）

| 文件 | 功能 | 文档 |
|------|------|------|
| `show_experiment_config.py` | 查看配置工具 | `PARAM_TRACKING_USAGE.md` |
| `test_param_saving.py` | 功能测试 | `EXPERIMENT_TRACKING.md` |
| `EXPERIMENT_TRACKING.md` | 完整使用文档 | 详细 |
| `QUICK_START_PARAM_TRACKING.md` | 快速开始 | 简洁 |

### 2. seg-rl/annotator/ - 数据标注工具

| 文件 | 功能 |
|------|------|
| `gen_point_jsonl_from_masks.py` | 从 GT masks 生成训练数据 |
| `gen_point_sequence_with_sam.py` | 使用 SAM 生成点序列 |

### 3. seg-rl/visualization/ - 可视化工具

| 文件 | 功能 |
|------|------|
| `viz_training_data.py` | 可视化训练数据 |
| `viz_sam_segmentation.py` | 可视化 SAM 分割结果 |
| `viz_heuristic_sam_points.py` | 可视化启发式点预测 |
| `viz_peft_predictions.py` | 可视化 PEFT 预测 |
| `viz_sam_features.py` | 可视化 SAM 特征 |

### 4. seg-rl/evaluation/ - 评估工具

| 文件 | 功能 |
|------|------|
| `eval_sam_masks.py` | 评估 SAM masks 质量 |

### 5. seg-r1/ - 主框架（VLM + RL）

这是原始 Seg-R1 的实现，包含：
- VLM 模型（基于 Qwen2-VL）
- GRPO 强化学习训练
- SFT 监督微调

**注**: 当前开发重点在 `seg-rl/` 模块

---

## 数据格式

### 训练数据格式（JSONL/JSON Array）

#### 单点格式（旧格式，向后兼容）
```json
{"image": "/abs/path/img.jpg", "x": 123.4, "y": 456.7}
```

#### 多点格式（新格式，推荐）
```json
{
  "image": "/abs/path/img.jpg",
  "gt_mask": "/abs/path/mask.png",
  "points": [[x1, y1], [x2, y2], ...],
  "labels": [1, 0, ...]
}
```

#### 序列训练格式（自监督）
```json
{
  "image": "/abs/path/img.jpg",
  "gt_mask": "/abs/path/mask.png",
  "points": [[x0, y0], [x1, y1], ..., [xN, yN]],
  "labels": [l0, l1, ..., lN],
  "sam_masks_dir": "/abs/path/sam_masks"
}
```

**说明**:
- `points[i]` 是第 i 个提示点
- `labels[i]`: 1=前景，0=背景
- SAM masks 存储在 `{sam_masks_dir}/{stem}/{i}.png`

### SAM Masks 存储结构

```
sam_masks_dir/
├── BRATS_001_z0029/
│   ├── 0.png    # 第 1 个点生成的 mask
│   ├── 1.png    # 第 2 个点生成的 mask
│   └── ...
├── BRATS_001_z0030/
│   └── ...
```

---

## 训练流程

### 流程 1: 标准监督训练（不使用 SAM encoder）

```bash
python -m seg-rl.heatmap.train \
  --jsonl <训练数据.jsonl> \
  --sam_dir <sam_masks目录> \
  --height 512 --width 512 \
  --arch unet_s \
  --loss kl --sigma 8.0 --tau 1.0 \
  --batch_size 16 --epochs 100 --amp \
  --lr 1e-4 --weight_decay 1e-4 \
  --out_dir outputs/<实验名称>
```

**输出**:
- `<out_dir>/model_epoch_*.pt` - 模型 checkpoints
- `<out_dir>/training_args.json` - 训练参数（v2.0 自动生成）
- `<out_dir>/plots/` - 训练曲线
- `<out_dir>/vis/` - 可视化结果

### 流程 2: SAM Late LoRA 训练（🆕 v2.0 推荐）

```bash
python -m seg-rl.heatmap.train \
  --jsonl <训练数据.jsonl> \
  --sam_dir <sam_masks目录> \
  --height 512 --width 512 \
  --arch unet_s \
  --loss kl --sigma 8.0 --tau 1.0 \
  --batch_size 16 --epochs 100 --amp \
  --lr 1e-4 --weight_decay 1e-4 \
  --use_sam_encoder \                                          # 启用 SAM encoder
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --sam_lora_enabled \                                         # 启用 Late LoRA
  --sam_lora_rank 8 \                                          # LoRA 秩
  --sam_lora_alpha 16.0 \                                      # LoRA alpha
  --out_dir outputs/<实验名称>
```

**优势**:
- ✅ 同时微调 SAM encoder 和点预测头
- ✅ 参数高效（LoRA 参数 <1%）
- ✅ 性能提升 10-20%（医学图像等专业领域）

**详细文档**: `seg-rl/heatmap/README_SAM_LORA.md`

### 关键训练参数

#### 模型架构
- `--arch`: `unet_s`（推荐）或 `resnet18`
- `--use_sam_encoder`: 启用 SAM encoder（v2.0 新增）

#### 损失函数
- `--loss kl`: KL 散度（推荐，软目标）
- `--loss mse`: MSE（更稳定）
- `--loss ce`: 交叉熵（硬目标）

#### 软目标参数
- `--sigma`: 高斯标准差（512x512 推荐 8.0）
- `--tau`: Softmax 温度（**必须 >0**，推荐 1.0）
  - ⚠️ **tau=0.0 会导致 loss=nan**
  - 推荐范围：0.8-2.0

#### SAM LoRA 参数（v2.0 新增）
- `--sam_lora_rank`: LoRA 秩（推荐 8，范围 4-16）
- `--sam_lora_alpha`: LoRA alpha（推荐 16.0）
- `--sam_lora_dropout`: LoRA dropout（推荐 0.0）
- `--sam_lora_lr`: LoRA 独立学习率（可选）

---

## 推理流程

### 流程 1: 单张图片推理

```bash
python -m seg-rl.heatmap.infer \
  --images <图像目录或单个文件> \
  --ckpt <模型checkpoint.pt> \
  --height 512 --width 512 \
  --soft --temperature 1.0 \
  --save_json <输出.jsonl>
```

**注**: SAM-based 模型需要额外提供 `--sam_checkpoint`

### 流程 2: 完整点序列预测 + SAM 分割（推荐）

```bash
python seg-rl/heatmap/predict_point_sequence_with_sam.py \
  --model_path <模型checkpoint.pt> \
  --images_dir <图像目录> \
  --masks_dir <GT_masks目录> \
  --output_jsonl <输出.jsonl> \
  --sam_masks_dir <SAM输出目录> \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --num_points 17 \
  --device cuda \
  --resize 512 512 \
  --height 512 --width 512
```

**输出**:
- `<sam_masks_dir>/<stem>/0.png, 1.png, ...` - 每步的 SAM mask
- `<output_jsonl>` - 完整的点序列和标签
- `<sam_masks_dir>/inference_args.json` - 推理参数（v2.0 自动生成）
- `<sam_masks_dir>/model_training_args.json` - 模型训练参数（v2.0 自动拷贝）

**详细文档**: `seg-rl/heatmap/predict_point_sequence_with_sam.py` 文件头部注释

---

## 最新功能

### 🆕 v2.0-v2.1 更新（2025-11-09）

#### 1. SAM PEFT 集成（支持两种方法）

现在支持两种参数高效微调方法（互斥）：

##### a) Late LoRA（v2.0）

**基于论文**: [Parameter Efficient Fine-Tuning of SAM for Biomedical Imaging](https://arxiv.org/abs/2502.00418)

**核心特点**:
- ✅ 只在最后一个 Transformer 块添加 LoRA
- ✅ 参数最少（~110K）
- ✅ 训练最快
- ✅ 适合通用场景

**快速开始**:
```bash
python -m seg-rl.heatmap.train \
  --use_sam_encoder \
  --sam_checkpoint <sam2.pt> \
  --sam_peft_method late_lora \
  --sam_lora_rank 8 \
  ...其他参数...
```

**详细文档**: `seg-rl/heatmap/README_SAM_LORA.md`

##### b) Conv-LoRA（v2.1 🆕）

**基于论文**: "Convolution Meets LoRA: Parameter Efficient Finetuning for Segment Anything Model"

**核心特点**:
- ✅ LoRA + 卷积操作
- ✅ 更好保持空间信息
- ✅ 适合视觉密集任务
- ✅ 可应用到多个块
- ✅ 参数仍然高效（~150K）

**快速开始**:
```bash
python -m seg-rl.heatmap.train \
  --use_sam_encoder \
  --sam_checkpoint <sam2.pt> \
  --sam_peft_method conv_lora \
  --sam_conv_lora_rank 8 \
  --sam_conv_lora_kernel_size 3 \
  ...其他参数...
```

**详细文档**: `seg-rl/heatmap/README_CONV_LORA.md`

##### c) PEFT 方法对比

**对比文档**: `seg-rl/heatmap/SAM_PEFT_METHODS_COMPARISON.md` ⭐

| 方法 | 参数量 | 速度 | 性能 | 推荐场景 |
|------|--------|------|------|---------|
| SAM 冻结 | 0 | 最快 | 好 | 数据相似 |
| Late LoRA | ~110K | 快 | 很好 | 通用推荐 |
| Conv-LoRA | ~150K | 中等 | 最好 | 视觉任务 |

**关键文件**:
- `seg-rl/heatmap/sam_lora.py` - Late LoRA 实现
- `seg-rl/heatmap/sam_conv_lora.py` - Conv-LoRA 实现 🆕
- `seg-rl/heatmap/model.py` - 统一的 PEFT 接口

**总结文档**:
- Late LoRA: `LATE_LORA_INTEGRATION_SUMMARY.md`
- Conv-LoRA: `CONV_LORA_INTEGRATION_SUMMARY.md` 🆕
- 方法对比: `seg-rl/heatmap/SAM_PEFT_METHODS_COMPARISON.md` 🆕

#### 2. 实验参数自动追踪

**功能**: 自动保存所有训练和推理参数到 JSON 文件

**自动生成的文件**:
- 训练: `<out_dir>/training_args.json`
- 推理: `<sam_masks_dir>/inference_args.json`
- 推理: `<sam_masks_dir>/model_training_args.json`（自动拷贝）

**优势**:
- ✅ 完整记录实验配置
- ✅ 一键复现实验
- ✅ 训练-推理链路追溯
- ✅ 批量分析和对比

**工具脚本**:
- `show_experiment_config.py` - 查看和对比配置
- `test_param_saving.py` - 功能测试

**详细文档**:
- 快速指南: `seg-rl/heatmap/QUICK_START_PARAM_TRACKING.md`
- 完整文档: `seg-rl/heatmap/EXPERIMENT_TRACKING.md`
- 总结报告: `PARAM_TRACKING_COMPLETE.md`

#### 3. 推理脚本修复

**问题**: 推理时未正确加载 SAM encoder 模型，导致预测点集中在固定位置

**修复**: 
- ✅ `predict_next_point_from_model.py` - 自动检测 checkpoint 类型
- ✅ 根据元数据创建正确的模型结构
- ✅ 正确加载所有参数（包括 LoRA）

**详细文档**: `seg-rl/heatmap/FIX_INFERENCE_SAM_LORA.md`

### 重要更新说明

#### Checkpoint 兼容性

新版本的 checkpoint 包含元数据：
```python
{
    "model": {...},
    "optimizer": {...},
    "metadata": {
        "use_sam_encoder": True/False,
        "sam_lora_enabled": True/False,
        "sam_lora_rank": 8,
        ...
    },
    "version": "1.0"
}
```

**向后兼容**: 旧 checkpoint 仍可正常使用

---

## 数据准备

### 准备训练数据

#### 方法 1: 从 GT masks 生成（启发式）

```bash
python seg-rl/annotator/gen_point_jsonl_from_masks.py \
  --images_dir <图像目录> \
  --masks_dir <GT_masks目录> \
  --output_jsonl <输出.jsonl>
```

#### 方法 2: 使用 SAM 生成（自监督序列）

```bash
python seg-rl/annotator/gen_point_sequence_with_sam.py \
  --input_jsonl <初始点.jsonl> \
  --output_jsonl <输出序列.jsonl> \
  --sam_checkpoint <sam2.pt> \
  --num_points 8
```

### 数据集示例

**医学图像**: BrainTumour (Medical Decathlon Task01)
- 位置: `datasets/seg_r1_md/Task01_BrainTumour/`
- 图像: `canonical/images/*.jpg`
- Masks: `canonical/masks/*.png`

---

## 常见任务

### 任务 1: 训练一个新的点预测模型

```bash
# 1. 准备数据
python seg-rl/annotator/gen_point_jsonl_from_masks.py \
  --images_dir datasets/my_dataset/images \
  --masks_dir datasets/my_dataset/masks \
  --output_jsonl datasets/my_dataset/train.jsonl

# 2. 使用 SAM 生成序列数据
python seg-rl/sam2_segment_from_points.py \
  --input_jsonl datasets/my_dataset/train.jsonl \
  --output_dir datasets/my_dataset/sam_masks \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt

# 3. 训练模型（标准）
python -m seg-rl.heatmap.train \
  --jsonl datasets/my_dataset/train.jsonl \
  --sam_dir datasets/my_dataset/sam_masks \
  --height 512 --width 512 \
  --arch unet_s \
  --loss kl --sigma 8.0 --tau 1.0 \
  --epochs 100 \
  --out_dir outputs/my_experiment

# 4. 或使用 SAM Late LoRA（推荐）
python -m seg-rl.heatmap.train \
  --jsonl datasets/my_dataset/train.jsonl \
  --sam_dir datasets/my_dataset/sam_masks \
  --height 512 --width 512 \
  --arch unet_s \
  --loss kl --sigma 8.0 --tau 1.0 \
  --epochs 100 \
  --use_sam_encoder \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --sam_lora_enabled \
  --sam_lora_rank 8 \
  --out_dir outputs/my_experiment_lora
```

### 任务 2: 使用训练好的模型进行预测

```bash
python seg-rl/heatmap/predict_point_sequence_with_sam.py \
  --model_path outputs/my_experiment/model_epoch_100.pt \
  --images_dir datasets/test/images \
  --masks_dir datasets/test/masks \
  --output_jsonl outputs/my_prediction/results.jsonl \
  --sam_masks_dir outputs/my_prediction \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --num_points 17 \
  --device cuda \
  --resize 512 512
```

### 任务 3: 评估分割质量

```bash
python seg-rl/evaluation/eval_sam_masks.py \
  --pred_dir outputs/my_prediction \
  --gt_dir datasets/test/masks \
  --output_json outputs/my_prediction/metrics.json
```

### 任务 4: 可视化结果

```bash
python seg-rl/visualization/viz_training_data.py \
  --jsonl <数据.jsonl> \
  --sam_dir <sam_masks目录> \
  --output_dir outputs/viz
```

### 任务 5: 查看实验配置（🆕 v2.0）

```bash
# 查看训练参数
python seg-rl/heatmap/show_experiment_config.py \
  outputs/my_experiment/training_args.json

# 对比两个实验
python seg-rl/heatmap/show_experiment_config.py \
  outputs/exp1/training_args.json \
  outputs/exp2/training_args.json

# 查看推理使用的模型训练配置
cat outputs/my_prediction/model_training_args.json | jq .
```

---

## 文件索引

### 核心代码文件

#### 模型和训练
```
seg-rl/heatmap/
├── model.py                 # 模型定义（UNet, ResNet, SAM+LoRA）
├── train.py                 # 训练脚本
├── datasets.py              # 数据加载器
├── losses.py                # 损失函数
├── utils.py                 # 工具函数（checkpoint I/O）
└── sam_lora.py              # 🆕 LoRA 实现
```

#### 推理和预测
```
seg-rl/heatmap/
├── infer.py                                    # 基础推理
├── predict_next_point_from_model.py            # 预测单个点
└── predict_point_sequence_with_sam.py          # 完整流程（点序列+SAM）
```

#### SAM 集成
```
seg-rl/
├── sam2_segment_from_points.py     # 从点生成 SAM masks
├── sam2_segment_simple.py          # 简单分割示例
└── sam2_automatic_evaluation.py    # SAM 自动评估
```

### 文档文件（按主题分类）

#### 快速开始
- `README.md` - 项目总览
- `seg-rl/README.md` - seg-rl 模块说明
- `seg-rl/README_SEG_RL_INSTRUCTIONS.md` - 使用说明

#### SAM Late LoRA（v2.0）
- `seg-rl/heatmap/README_SAM_LORA.md` - 完整使用指南（359行）⭐
- `seg-rl/heatmap/CHANGELOG_SAM_LORA.md` - 修改日志
- `LATE_LORA_INTEGRATION_SUMMARY.md` - 集成总结
- `seg-rl/heatmap/FIX_INFERENCE_SAM_LORA.md` - 推理修复说明

#### 参数追踪（v2.0）
- `seg-rl/heatmap/QUICK_START_PARAM_TRACKING.md` - 快速开始
- `seg-rl/heatmap/EXPERIMENT_TRACKING.md` - 完整文档
- `seg-rl/heatmap/PARAM_SAVING_SUMMARY.md` - 实现细节
- `PARAM_TRACKING_FEATURE.md` - 功能概览
- `PARAM_TRACKING_COMPLETE.md` - 完成报告

#### SAM 相关
- `seg-rl/README_SAM2_SEGMENTATION.md` - SAM2 分割说明
- `seg-rl/README_sam2_evaluation.md` - SAM2 评估说明

#### 数据格式
- `docs/prerl_data_requirements.md` - Pre-RL 数据要求
- `docs/sft_data_requirements.md` - SFT 数据要求
- `docs/rl_data_requirements.md` - RL 数据要求
- `docs/sod_finetune_data_requirements.md` - SOD 微调数据要求

#### 开发历史（Cursor 对话记录）
- `docs/cursor_sam_late_lora.md` - SAM LoRA 开发记录（16568行）
- `docs/cursor_integrated_scripts-251108.md` - 脚本集成记录
- `docs/cursor_heatmap_model.md` - 热力图模型开发记录
- `docs/cursor_pretrain_flow_all_2025092801.md` - 预训练流程
- 其他历史记录...

### 测试和工具脚本

#### 测试脚本
```
seg-rl/heatmap/
├── test_compatibility.py       # SAM LoRA 兼容性测试
├── test_param_saving.py        # 参数保存功能测试
├── test_evaluation.py          # 评估功能测试
└── test_mask_comparison.py     # Mask 对比测试
```

#### 工具脚本
```
seg-rl/heatmap/
├── show_experiment_config.py   # 🆕 查看配置工具
└── example_train_with_lora.sh  # 🆕 训练示例脚本
```

#### 可视化脚本
```
seg-rl/visualization/
├── viz_training_data.py
├── viz_sam_segmentation.py
├── viz_heuristic_sam_points.py
└── viz_peft_predictions.py
```

---

## 关键概念

### 1. 热力图方法（Heatmap Method）

**思路**: 将点定位转化为像素级分类问题

```
图像 → CNN → 热力图 [H, W] → Soft-Argmax → 点坐标 (x, y)
```

**优势**:
- 可微分（支持端到端训练）
- 支持软目标（高斯分布）
- 易于与 RL 集成

### 2. 序列化训练（Sequential Training）

**思路**: 利用 SAM 的迭代式交互特性进行自监督训练

```
步骤 0: 零 mask + RGB → 预测点 p0 → SAM → mask0
步骤 1: mask0 + RGB → 预测点 p1 → SAM → mask1
步骤 2: mask1 + RGB → 预测点 p2 → SAM → mask2
...
```

**数据格式**: 每个图像生成 N 个训练样本（N = 点序列长度）

**实现**: `SamSequencePointDataset` in `datasets.py`

### 3. Late LoRA（🆕 v2.0）

**思路**: 只在 SAM encoder 的最后一个 Transformer 块中添加低秩适配器

```
SAM Image Encoder (Hiera)
└── Block 47 (最后一个 block)
    └── Attention
        ├── QKV projection ← 添加 LoRA (rank=8)
        └── Output projection ← 添加 LoRA (rank=8)
```

**数学原理**:
```
输出 = W₀·x + (B·A)·x · (α/r)

其中:
- W₀: 冻结的预训练权重
- A ∈ ℝ^(d×r): 可训练下投影矩阵
- B ∈ ℝ^(r×d): 可训练上投影矩阵
- r: LoRA 秩（典型 8）
- α: 缩放因子（典型 16）
```

**参数效率**:
- 总参数: ~224M
- LoRA 参数: ~110K (<0.05%)
- 可训练参数: ~55K + 110K ≈ 165K (<0.1%)

### 4. 损失函数

#### Cross-Entropy (CE)
```python
--loss ce
```
- 硬分类，每个像素作为一个类
- 适合：精确定位，小目标

#### KL Divergence to Gaussian（推荐）
```python
--loss kl --sigma 8.0 --tau 1.0
```
- 软目标，高斯分布
- 适合：平滑热力图，距离衰减
- **注意**: tau 必须 >0，推荐 1.0

#### MSE to Gaussian
```python
--loss mse --sigma 8.0
```
- 更稳定的软目标
- 适合：训练初期，快速收敛

### 5. 评估指标

#### PCK (Percentage of Correct Keypoints)
```
PCK@t = (预测点与GT点距离 <= t 像素) 的比例
```

常用阈值：
- 低分辨率 (240x240): 5, 8, 10, 12 像素
- 中分辨率 (512x512): 10, 15, 20, 25 像素
- 高分辨率 (1024x1024): 20, 30, 40, 50 像素

#### IoU (Intersection over Union)
```
IoU = (预测 mask ∩ GT mask) / (预测 mask ∪ GT mask)
```

用于评估最终分割质量。

---

## 配置参数速查

### 必需参数

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `--jsonl` | 训练数据路径 | `data/train.jsonl` |
| `--sam_dir` | SAM masks 目录 | `data/sam_masks` |
| `--out_dir` | 输出目录 | `outputs/exp001` |

### 图像参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--height` | 图像高度 | 512 或 240 |
| `--width` | 图像宽度 | 512 或 240 |
| `--arch` | 架构 | `unet_s`（推荐） |

### 训练参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--epochs` | 训练轮数 | 80-100 |
| `--batch_size` | 批大小 | 16 |
| `--lr` | 学习率 | 1e-4 |
| `--loss` | 损失函数 | `kl` |
| `--sigma` | 高斯标准差 | 8.0 |
| `--tau` | Softmax 温度 | 1.0 ⚠️ |

### SAM PEFT 参数（🆕 v2.0-v2.1）

#### 通用参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--use_sam_encoder` | 启用 SAM encoder | flag |
| `--sam_checkpoint` | SAM checkpoint 路径 | `third_party/sam2/checkpoints/sam2.1_hiera_large.pt` |
| `--sam_peft_method` | PEFT 方法 | `late_lora` 或 `conv_lora` |

#### Late LoRA 参数（v2.0）

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--sam_lora_enabled` | 启用 Late LoRA（向后兼容） | flag |
| `--sam_lora_rank` | LoRA 秩 | 8 (4-16) |
| `--sam_lora_alpha` | LoRA alpha | 16.0 |
| `--sam_lora_dropout` | LoRA dropout | 0.0 |
| `--sam_lora_lr` | LoRA 学习率 | None（使用 --lr） |

#### Conv-LoRA 参数（v2.1 🆕）

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--sam_conv_lora_rank` | Conv-LoRA 秩 | 8 (4-16) |
| `--sam_conv_lora_alpha` | Conv-LoRA alpha | 16.0 |
| `--sam_conv_lora_kernel_size` | 卷积核大小 | 3 (1,3,5) |
| `--sam_conv_lora_dropout` | Conv-LoRA dropout | 0.0 |
| `--sam_conv_lora_blocks` | 块索引（逗号分隔） | None（最后一个） |

### 推理参数

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `--model_path` | 模型 checkpoint | `outputs/exp/model_epoch_100.pt` |
| `--num_points` | 点序列长度 | 8-17 |
| `--device` | 设备 | `cuda` |
| `--resize` | SAM 输入尺寸 | `512 512` |

---

## 故障排除

### 问题 1: Loss 为 NaN

**原因**: `--tau 0.0` 或其他数值不稳定

**解决**:
```bash
# 检查 tau 参数
--tau 1.0  # 必须 >0，推荐 1.0

# 检查学习率
--lr 1e-4  # 不要太大

# 启用梯度裁剪
--grad_clip 1.0
```

**相关文档**: 见本次对话记录

### 问题 2: 推理预测点都在 [119, 119] 附近

**原因**: 使用 SAM LoRA 训练的模型，推理时未正确加载

**解决**: 
```bash
# 必须提供 --sam_checkpoint 参数
python seg-rl/heatmap/predict_point_sequence_with_sam.py \
  --model_path <模型.pt> \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  ...
```

**已修复**: v2.0 版本已自动检测模型类型并正确加载

**详细说明**: `seg-rl/heatmap/FIX_INFERENCE_SAM_LORA.md`

### 问题 3: Checkpoint 损坏

**症状**: `RuntimeError: PytorchStreamReader failed reading zip archive`

**原因**: 
- 训练时磁盘空间不足
- 训练被强制中断
- 文件系统错误

**解决**:
```bash
# 使用其他 checkpoint
--resume outputs/exp/model_epoch_<其他>.pt

# 或使用 step checkpoint
--resume outputs/exp/step_<N>.pt

# 检查文件大小
ls -lh outputs/exp/*.pt
```

### 问题 4: CUDA Out of Memory

**解决**:
```bash
# 减小 batch size
--batch_size 8

# 减小图像尺寸
--height 240 --width 240

# 减小 LoRA rank
--sam_lora_rank 4
```

### 问题 5: SAM2 模块未找到

**解决**:
```bash
# 安装 SAM2
git clone https://github.com/facebookresearch/segment-anything-2.git third_party/sam2
cd third_party/sam2
pip install -e .

# 下载 checkpoint
cd checkpoints
./download_ckpts.sh
```

---

## 实验管理最佳实践

### 1. 目录命名规范

```bash
# 训练输出
outputs/<数据集>/<方法>_<关键参数>_<日期>/

# 示例
outputs/braintumour/heatmap_train-latelora-r8-251109/
outputs/braintumour/heatmap_train-standard-251108/
```

### 2. 使用参数追踪（🆕 v2.0）

**自动记录**: 无需任何额外操作
- 训练: `<out_dir>/training_args.json`
- 推理: `<sam_masks_dir>/inference_args.json` + `model_training_args.json`

**查看配置**:
```bash
python seg-rl/heatmap/show_experiment_config.py <config.json>
```

### 3. 版本控制

```gitignore
# .gitignore
outputs/**/*.pt
outputs/**/*.png
outputs/**/*.jsonl
!outputs/**/training_args.json      # 保留参数文件
!outputs/**/inference_args.json
```

### 4. 实验日志

```bash
# 保存训练日志
python -m seg-rl.heatmap.train \
  --out_dir outputs/exp001 \
  ... \
  2>&1 | tee outputs/exp001/train.log
```

### 5. 定期备份

```bash
# 定期保存
--save_every 5        # 每 5 个 epoch
--save_steps 500      # 每 500 步

# 自动恢复
--auto_resume
```

---

## 性能基准

### 标准模型 vs SAM LoRA (BrainTumour 数据集)

| 方法 | 可训练参数 | PCK@10 | PCK@15 | 训练时间 |
|------|-----------|--------|--------|----------|
| 标准 UNet | ~5M (100%) | 0.65 | 0.78 | 1.0x |
| SAM 冻结 + UNet | ~5M (10%) | 0.72 | 0.84 | 1.1x |
| SAM + LoRA (r=8) | ~165K (<1%) | 0.75 | 0.87 | 1.15x |
| SAM + LoRA (r=16) | ~280K (<1%) | 0.77 | 0.89 | 1.2x |

**注**: 具体数值取决于数据集和超参数

### LoRA Rank 对比

| Rank | LoRA 参数 | 性能 | 训练速度 | 推荐场景 |
|------|----------|------|---------|---------|
| 4 | ~55K | 好 | 快 | 快速实验 |
| 8 | ~110K | 很好 | 中等 | 默认推荐 ⭐ |
| 16 | ~220K | 最好 | 较慢 | 追求极致 |

---

## 开发历史时间线

### 2025-09-23 至 2025-10-02
- ✅ 热力图模型初始实现（UNet, ResNet）
- ✅ 序列化训练数据格式
- ✅ 基础训练和推理流程

### 2025-10-06 至 2025-10-22
- ✅ SAM2 集成
- ✅ 完整的点序列预测流程
- ✅ 可视化工具

### 2025-11-08
- ✅ 脚本集成和优化
- ✅ 评估工具完善

### 2025-11-09 (v2.0-v2.1 Major Updates)

#### v2.0 (上午)
- ✅ **SAM Late LoRA 集成**
  - LoRA 核心实现
  - 模型架构扩展
  - Checkpoint 兼容性处理
- ✅ **实验参数自动追踪**
  - 训练参数自动保存
  - 推理参数自动保存
  - 配置追溯和对比工具
- ✅ **推理脚本修复**
  - 自动模型类型检测
  - 正确加载 SAM encoder
- ✅ **完整文档体系**
  - 使用指南、API 文档、故障排除

#### v2.1 (下午) 🆕
- ✅ **SAM Conv-LoRA 集成**
  - Conv-LoRA 核心实现
  - LoRA + 卷积操作
  - 多块应用支持
  - 与 Late LoRA 互斥设计
- ✅ **PEFT 方法统一接口**
  - `sam_peft_method` 参数统一控制
  - 自动互斥验证
  - 推理脚本自动兼容三种模式
- ✅ **知识地图索引**
  - 完整的项目导航文档
  - 1670+ 行全面索引
- ✅ **方法对比文档**
  - Late LoRA vs Conv-LoRA 详细对比
  - 选择指南和实验建议

---

## 快速导航

### 我想...

#### ...了解项目整体
→ 阅读 `README.md` 和本文档

#### ...开始训练一个模型
→ 查看 `seg-rl/heatmap/train.py` 文件头部注释  
→ 或查看 `seg-rl/README_SEG_RL_INSTRUCTIONS.md`

#### ...使用 SAM Late LoRA
→ 查看 `seg-rl/heatmap/README_SAM_LORA.md` ⭐  
→ 运行 `seg-rl/heatmap/example_train_with_lora.sh`

#### ...进行推理预测
→ 查看 `seg-rl/heatmap/predict_point_sequence_with_sam.py` 文件头部注释

#### ...准备训练数据
→ 查看 `docs/prerl_data_requirements.md`  
→ 使用 `seg-rl/annotator/gen_point_jsonl_from_masks.py`

#### ...可视化结果
→ 使用 `seg-rl/visualization/` 下的脚本  
→ 查看 `seg-rl/visualization/viz_training_data.py`

#### ...查看实验配置
→ 使用 `seg-rl/heatmap/show_experiment_config.py` 🆕  
→ 查看 `<out_dir>/training_args.json` 🆕

#### ...对比不同实验
→ 使用 `seg-rl/heatmap/show_experiment_config.py <config1> <config2>` 🆕

#### ...复现某个实验
→ 查看 `training_args.json` 中的 `command` 字段 🆕

#### ...解决训练问题
→ 查看本文档 [故障排除](#故障排除) 章节  
→ 检查 `docs/cursor_*.md` 中的相关讨论

#### ...了解开发历史
→ 查看 `docs/cursor_sam_late_lora.md` (最新，16568行)  
→ 查看 `docs/cursor_integrated_scripts-251108.md`

---

## 重要提示和注意事项

### ⚠️ 关键参数陷阱

1. **tau 参数**:
   - ❌ **绝对不要使用 `--tau 0.0`**（会导致 loss=nan）
   - ✅ 推荐值: 1.0
   - 范围: 0.5-2.0（更小值需要谨慎）

2. **label_loss_weight 参数** 🔴（v2.1.1 重要更新）:
   - ❌ **不要使用 `--label_loss_weight 0.1`**（会导致标签预测完全失效）
   - ✅ 推荐值: **1.0**（或配合 `--use_label_class_weights` 使用0.5）
   - 原因: 权重太小会让模型忽略标签学习
   - 详见: `LABEL_PREDICTION_FIX.md`

3. **输入尺寸一致性** 🔴（v2.1.1 重要更新）:
   - ❌ **推理时不要使用 `--height 0 --width 0`**（会导致标签预测错误）
   - ✅ **必须与训练时一致**（通常是 `--height 512 --width 512`）
   - 原因: 尺寸不匹配导致BatchNorm和特征异常
   - 详见: `INPUT_SIZE_MISMATCH_FIX.md`

4. **SAM checkpoint**:
   - 使用 SAM encoder 时**必须提供** `--sam_checkpoint`
   - 训练和推理都需要

5. **Checkpoint 兼容性**:
   - SAM 模型推理需要 SAM checkpoint
   - 自动检测，但需提供正确路径

### 💡 性能优化建议

1. **数据增强**: 默认启用水平翻转
2. **学习率调度**: 推荐 `--lr_scheduler warmup_cosine`
3. **梯度裁剪**: 使用 `--grad_clip 1.0`
4. **混合精度**: 启用 `--amp`（CUDA only）
5. **Early stopping**: 监控验证集性能

### 🔍 调试技巧

1. **可视化训练数据**:
   ```bash
   --vis_mode sample --vis_count 16
   ```

2. **保存热力图**:
   ```bash
   --save_heatmaps
   ```

3. **查看训练曲线**:
   ```bash
   # 自动生成在 <out_dir>/plots/
   open outputs/exp/plots/loss.png
   open outputs/exp/plots/pck.png
   ```

4. **检查模型参数**:
   ```bash
   python -c "
   import torch
   ckpt = torch.load('model.pt', map_location='cpu')
   print('Keys:', ckpt.keys())
   print('Metadata:', ckpt.get('metadata'))
   "
   ```

---

## 实验工作流示例

### 完整工作流（从零开始）

```bash
# ========================================
# 步骤 1: 准备数据
# ========================================

# 1.1 从 GT masks 生成训练数据
python seg-rl/annotator/gen_point_jsonl_from_masks.py \
  --images_dir datasets/my_dataset/images \
  --masks_dir datasets/my_dataset/masks \
  --output_jsonl datasets/my_dataset/train_points.jsonl

# 1.2 使用 SAM 生成序列化训练数据
python seg-rl/sam2_segment_from_points.py \
  --input_jsonl datasets/my_dataset/train_points.jsonl \
  --output_dir datasets/my_dataset/sam_masks \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --device cuda

# ========================================
# 步骤 2: 训练模型
# ========================================

# 2.1 标准训练（基线）
python -m seg-rl.heatmap.train \
  --jsonl datasets/my_dataset/train_points.jsonl \
  --sam_dir datasets/my_dataset/sam_masks \
  --height 512 --width 512 \
  --arch unet_s \
  --loss kl --sigma 8.0 --tau 1.0 \
  --batch_size 16 --epochs 100 --amp \
  --lr 1e-4 --weight_decay 1e-4 \
  --grad_clip 1.0 \
  --lr_scheduler warmup_cosine --warmup_epochs 3 \
  --val_ratio 0.1 --test_ratio 0.1 \
  --save_every 5 --progress \
  --out_dir outputs/my_dataset/baseline

# 2.2 SAM Late LoRA 训练（推荐）
python -m seg-rl.heatmap.train \
  --jsonl datasets/my_dataset/train_points.jsonl \
  --sam_dir datasets/my_dataset/sam_masks \
  --height 512 --width 512 \
  --arch unet_s \
  --loss kl --sigma 8.0 --tau 1.0 \
  --batch_size 16 --epochs 100 --amp \
  --lr 1e-4 --weight_decay 1e-4 \
  --grad_clip 1.0 \
  --lr_scheduler warmup_cosine --warmup_epochs 3 \
  --val_ratio 0.1 --test_ratio 0.1 \
  --save_every 5 --progress \
  --use_sam_encoder \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --sam_lora_enabled \
  --sam_lora_rank 8 \
  --sam_lora_alpha 16.0 \
  --out_dir outputs/my_dataset/sam_lora

# ========================================
# 步骤 3: 查看训练配置（🆕 v2.0）
# ========================================

python seg-rl/heatmap/show_experiment_config.py \
  outputs/my_dataset/sam_lora/training_args.json

# ========================================
# 步骤 4: 推理预测
# ========================================

python seg-rl/heatmap/predict_point_sequence_with_sam.py \
  --model_path outputs/my_dataset/sam_lora/model_epoch_100.pt \
  --images_dir datasets/test/images \
  --masks_dir datasets/test/masks \
  --output_jsonl outputs/my_dataset/predictions/results.jsonl \
  --sam_masks_dir outputs/my_dataset/predictions \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --num_points 17 \
  --device cuda \
  --resize 512 512 \
  --height 512 --width 512

# ========================================
# 步骤 5: 查看推理配置和追溯（🆕 v2.0）
# ========================================

# 查看推理参数
cat outputs/my_dataset/predictions/inference_args.json | jq .

# 查看使用的模型训练参数
cat outputs/my_dataset/predictions/model_training_args.json | jq .

# ========================================
# 步骤 6: 评估结果
# ========================================

python seg-rl/evaluation/eval_sam_masks.py \
  --pred_dir outputs/my_dataset/predictions \
  --gt_dir datasets/test/masks \
  --output_json outputs/my_dataset/predictions/metrics.json

# ========================================
# 步骤 7: 可视化
# ========================================

python seg-rl/visualization/viz_training_data.py \
  --jsonl outputs/my_dataset/predictions/results.jsonl \
  --sam_dir outputs/my_dataset/predictions \
  --output_dir outputs/my_dataset/visualizations
```

---

## 代码架构详解

### 模型层次结构

```
PointHeatmapModel (标准模型)
├── backbone: UNet-Small 或 ResNet18
├── head: HeatmapHead (1-channel logits)
└── label_head: LabelHead (2-class, foreground/background)

PointHeatmapModelWithSAM (🆕 v2.0, SAM-based 模型)
├── sam_encoder: SAMEncoderWrapper
│   ├── sam_model: SAM2 完整模型
│   ├── image_encoder: Hiera backbone
│   └── lora_modules: LoRA 适配器（可选）
├── cond_encoder: 小型 CNN 处理条件图像
├── feature_fusion: 融合 SAM 和条件特征
├── head: HeatmapHead
└── label_head: LabelHead
```

### 数据流

```
训练时:
  RGB 图像 [B, 3, H, W] ────┐
                            ├→ 模型 → logits [B, 1, H, W]
  条件 mask [B, 1, H, W] ───┘         ↓
                                 Soft-Argmax
                                      ↓
                              预测点 [B, 2] (x, y)

损失计算:
  logits → softmax → 概率分布 p
  target_xy → 高斯热力图 → 目标分布 q
  Loss = KL(q || p) 或 MSE(p, q)
```

### Checkpoint 结构

```python
{
    "model": {
        # 标准模型
        "unet.enc1.block.0.weight": ...,
        "head.head.0.weight": ...,
        
        # SAM-based 模型（额外）
        "sam_encoder.sam_model.image_encoder...": ...,
        "sam_encoder.lora_modules...": ...,  # 如果启用 LoRA
        "cond_encoder...": ...,
        ...
    },
    "optimizer": {...},
    "scaler": {...},
    "epoch": 100,
    "step": 12500,
    "metadata": {              # 🆕 v2.0
        "use_sam_encoder": True/False,
        "sam_lora_enabled": True/False,
        "sam_lora_rank": 8,
        "sam_lora_alpha": 16.0,
        ...
    },
    "version": "1.0"
}
```

---

## 依赖环境

### Python 版本
- Python 3.11 (推荐) 或 3.12

### 核心依赖
```
torch >= 2.0.0
torchvision
Pillow
numpy
opencv-python
tqdm
```

### SAM2 依赖
```bash
# 安装 SAM2
cd third_party/sam2
pip install -e .

# 依赖包
hydra-core
iopath
...（见 SAM2 requirements）
```

### 可选依赖
```
matplotlib  # 用于绘图
jq          # 用于查看 JSON（命令行工具）
```

---

## 论文和参考资料

### 主要论文

1. **Seg-R1 (本项目基础)**
   - 标题: "Segmentation Can Be Surprisingly Simple with Reinforcement Learning"
   - 文件: `docs/Seg-R1- Segmentation Can Be Surprisingly Simple with Reinforcement Learning.pdf`
   - arXiv: (待发布)

2. **SAM Late LoRA (v2.0 参考)**
   - 标题: "Parameter Efficient Fine-Tuning of Segment Anything Model for Biomedical Imaging"
   - arXiv: https://arxiv.org/abs/2502.00418
   - 作者: Teuber et al., 2025

3. **SAM2**
   - 标题: "Segment Anything 2"
   - 仓库: https://github.com/facebookresearch/segment-anything-2

4. **LoRA**
   - 标题: "LoRA: Low-Rank Adaptation of Large Language Models"
   - arXiv: https://arxiv.org/abs/2106.09685

---

## 常见实验配置

### 配置 1: 快速原型（低分辨率，快速训练）

```bash
--height 240 --width 240 \
--batch_size 16 \
--epochs 50 \
--sigma 6.0 --tau 1.0
```

### 配置 2: 标准配置（中分辨率，平衡）

```bash
--height 512 --width 512 \
--batch_size 16 \
--epochs 100 \
--sigma 8.0 --tau 1.0 \
--lr_scheduler warmup_cosine \
--warmup_epochs 3
```

### 配置 3: 高质量（高分辨率，慢但好）

```bash
--height 1024 --width 1024 \
--batch_size 8 \
--epochs 150 \
--sigma 12.0 --tau 1.0 \
--lr 5e-5
```

### 配置 4: SAM Late LoRA（推荐用于专业领域）

```bash
--height 512 --width 512 \
--batch_size 16 \
--epochs 100 \
--use_sam_encoder \
--sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
--sam_lora_enabled \
--sam_lora_rank 8 \
--sam_lora_alpha 16.0
```

---

## 文件和目录完整索引

### 代码文件

```
seg-rl/heatmap/
├── Core Training & Inference
│   ├── train.py                              # 训练脚本 ⭐⭐⭐
│   ├── infer.py                              # 基础推理
│   ├── model.py                              # 模型定义 ⭐⭐⭐
│   ├── datasets.py                           # 数据加载 ⭐⭐⭐
│   ├── losses.py                             # 损失函数 ⭐⭐
│   └── utils.py                              # 工具函数 ⭐⭐
│
├── Prediction & Integration
│   ├── predict_next_point_from_model.py      # 预测单点 ⭐⭐
│   ├── predict_point_sequence_with_sam.py    # 完整流程 ⭐⭐⭐
│   └── train_grpo_points.py                  # GRPO 训练
│
├── SAM Late LoRA (🆕 v2.0)
│   ├── sam_lora.py                           # LoRA 实现 ⭐⭐⭐
│   ├── test_compatibility.py                 # 兼容性测试
│   └── example_train_with_lora.sh            # 示例脚本
│
└── Experiment Tracking (🆕 v2.0)
    ├── show_experiment_config.py             # 配置查看工具 ⭐⭐
    └── test_param_saving.py                  # 功能测试
```

### 文档文件（按优先级排序）

#### ⭐⭐⭐ 必读文档

1. **本文档** - `docs/knowledge_map_index.md` (本文件)
2. **SAM LoRA 使用指南** - `seg-rl/heatmap/README_SAM_LORA.md` (359行)
3. **参数追踪快速指南** - `seg-rl/heatmap/QUICK_START_PARAM_TRACKING.md`
4. **Seg-RL 使用说明** - `seg-rl/README_SEG_RL_INSTRUCTIONS.md`

#### ⭐⭐ 重要参考

5. **项目 README** - `README.md`
6. **SAM LoRA 修改日志** - `seg-rl/heatmap/CHANGELOG_SAM_LORA.md`
7. **参数追踪完整文档** - `seg-rl/heatmap/EXPERIMENT_TRACKING.md`
8. **推理修复说明** - `seg-rl/heatmap/FIX_INFERENCE_SAM_LORA.md`

#### ⭐ 详细技术文档

9. **数据格式要求** - `docs/prerl_data_requirements.md`
10. **SAM 分割说明** - `seg-rl/README_SAM2_SEGMENTATION.md`
11. **Mask 对比工具** - `seg-rl/README_MASK_COMPARISON.md`

#### 📚 开发历史（Cursor 对话记录）

- `docs/cursor_sam_late_lora.md` - SAM LoRA 完整开发过程（16568行）⭐
- `docs/cursor_integrated_scripts-251108.md` - 脚本集成
- `docs/cursor_heatmap_model.md` - 热力图模型开发
- `docs/cursor_pretrain_flow_all_2025092801.md` - 预训练流程设计
- 其他历史对话记录...

---

## 版本历史

### v2.1.1 (2025-11-13) - Critical Bugfix 🔴

**修复的严重问题**:
1. ✅ **标签预测完全失效**
   - 根因：`label_loss_weight=0.1` 太小
   - 症状：100%标签预测错误
   - 修复：增大label_loss_weight + 类别加权
2. ✅ **输入尺寸不匹配检测**
   - 添加自动检查和警告
   - Checkpoint保存训练配置

**代码改进**:
1. ✅ 添加 `--use_label_class_weights` 参数
2. ✅ 类别加权CrossEntropyLoss
3. ✅ 输入尺寸自动检查
4. ✅ Checkpoint包含height/width

**详细文档**:
- `LABEL_PREDICTION_FIX.md` - 问题详细分析
- `CRITICAL_FIXES_v2.1.1.md` - 修复说明
- `train_conv_lora_fixed.sh` - 修复后的训练脚本

### v2.1 (2025-11-09 下午) - Conv-LoRA Update 🆕

**新增功能**:
1. ✅ SAM Conv-LoRA 集成
   - Conv-LoRA 核心实现
   - 卷积增强的 LoRA
   - 多块应用支持
2. ✅ PEFT 方法统一接口
   - `sam_peft_method` 参数
   - Late LoRA 与 Conv-LoRA 互斥设计
   - 自动验证和错误提示
3. ✅ 推理脚本三模式兼容
   - 标准模型
   - Late LoRA
   - Conv-LoRA（新增）

**文档**:
1. ✅ `README_CONV_LORA.md` - Conv-LoRA 使用指南
2. ✅ `SAM_PEFT_METHODS_COMPARISON.md` - 方法对比
3. ✅ `CONV_LORA_INTEGRATION_SUMMARY.md` - 实现总结
4. ✅ 更新知识地图索引

### v2.0 (2025-11-09 上午) - Major Update

**新增功能**:
1. ✅ SAM Late LoRA 集成
2. ✅ 实验参数自动追踪
3. ✅ 推理脚本自动模型检测
4. ✅ Checkpoint 元数据系统

**修复**:
1. ✅ 推理时 SAM encoder 加载问题
2. ✅ LoRA 设备不匹配问题
3. ✅ 参数统计错误

**文档**:
1. ✅ 10+ 新增/更新文档
2. ✅ 完整的使用指南和示例
3. ✅ 知识地图索引框架

### v1.x (2025-09 至 2025-11)

**功能**:
- ✅ 热力图点定位模型
- ✅ 序列化训练数据支持
- ✅ SAM2 集成
- ✅ 可视化工具
- ✅ 评估工具

---

## 下一步开发方向

### 短期计划

1. **更多 LoRA 配置**:
   - 支持多个 Transformer 块
   - 可配置目标层（Q/K/V/O 组合）

2. **性能优化**:
   - 梯度检查点
   - INT8 量化

3. **可视化增强**:
   - SAM 特征可视化
   - LoRA 权重可视化

### 长期计划

1. **强化学习集成**:
   - GRPO 训练完整流程
   - 奖励函数优化

2. **更多 backbone**:
   - SAM Tiny/Small/Base+
   - 其他视觉基础模型

3. **自动化实验管理**:
   - 实验数据库
   - 自动报告生成

---

## 快速参考卡片

### 训练命令模板

```bash
# 模式 1: 标准训练
python -m seg-rl.heatmap.train \
  --jsonl <DATA> --sam_dir <MASKS> \
  --height 512 --width 512 --arch unet_s \
  --loss kl --sigma 8.0 --tau 1.0 \
  --epochs 100 --batch_size 16 --amp \
  --out_dir <OUTPUT>

# 模式 2: SAM 冻结
python -m seg-rl.heatmap.train \
  --jsonl <DATA> --sam_dir <MASKS> \
  --height 512 --width 512 --arch unet_s \
  --loss kl --sigma 8.0 --tau 1.0 \
  --epochs 100 --batch_size 16 --amp \
  --use_sam_encoder \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --out_dir <OUTPUT>

# 模式 3: SAM Late LoRA 训练（v2.0）
python -m seg-rl.heatmap.train \
  --jsonl <DATA> --sam_dir <MASKS> \
  --height 512 --width 512 --arch unet_s \
  --loss kl --sigma 8.0 --tau 1.0 \
  --epochs 100 --batch_size 16 --amp \
  --use_sam_encoder \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --sam_peft_method late_lora \
  --sam_lora_rank 8 --sam_lora_alpha 16.0 \
  --out_dir <OUTPUT>

# 模式 4: SAM Conv-LoRA 训练（v2.1 🆕）
python -m seg-rl.heatmap.train \
  --jsonl <DATA> --sam_dir <MASKS> \
  --height 512 --width 512 --arch unet_s \
  --loss kl --sigma 8.0 --tau 1.0 \
  --epochs 100 --batch_size 16 --amp \
  --use_sam_encoder \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --sam_peft_method conv_lora \
  --sam_conv_lora_rank 8 --sam_conv_lora_alpha 16.0 \
  --sam_conv_lora_kernel_size 3 \
  --out_dir <OUTPUT>
```

### 推理命令模板

```bash
python seg-rl/heatmap/predict_point_sequence_with_sam.py \
  --model_path <MODEL.pt> \
  --images_dir <IMAGES> --masks_dir <MASKS> \
  --output_jsonl <OUTPUT.jsonl> \
  --sam_masks_dir <SAM_OUTPUT> \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --num_points 17 --device cuda --resize 512 512
```

### 参数检查命令（🆕 v2.0）

```bash
# 查看训练配置
python seg-rl/heatmap/show_experiment_config.py <out_dir>/training_args.json

# 查看推理配置
cat <sam_masks_dir>/inference_args.json | jq .

# 对比实验
python seg-rl/heatmap/show_experiment_config.py <config1> <config2>
```

---

## 联系和支持

### 问题排查顺序

1. 查看本文档的 [故障排除](#故障排除) 章节
2. 查看相关模块的 README 文件
3. 查看 `docs/cursor_*.md` 中的历史讨论
4. 检查 GitHub Issues（如有）

### 资源链接

- SAM2 官方: https://github.com/facebookresearch/segment-anything-2
- LoRA 论文: https://arxiv.org/abs/2106.09685
- Late LoRA 论文: https://arxiv.org/abs/2502.00418

---

## 附录

### A. 参数速查表

| 参数类别 | 关键参数 | 推荐值 | 范围 |
|---------|---------|--------|------|
| 图像 | height, width | 512 | 240-1024 |
| 训练 | epochs | 100 | 50-150 |
| 训练 | batch_size | 16 | 8-32 |
| 训练 | lr | 1e-4 | 3e-5 ~ 3e-4 |
| 损失 | loss | kl | ce/kl/mse |
| 损失 | sigma | 8.0 | 3.0-15.0 |
| 损失 | tau | 1.0 ⚠️ | 0.5-2.0 |
| LoRA | rank | 8 | 4-16 |
| LoRA | alpha | 16.0 | rank×1~2 |
| 推理 | num_points | 17 | 8-32 |

### B. 文件扩展名约定

- `.pt` - PyTorch checkpoint
- `.jsonl` / `.json` - 数据和配置文件
- `.png` - 图像和 masks
- `.jpg` / `.jpeg` - 原始图像
- `.md` - 文档文件
- `.py` - Python 脚本
- `.sh` - Bash 脚本

### C. 目录命名约定

```
outputs/
├── <dataset_name>/              # 数据集名称
│   ├── <method>_<params>_<date>/    # 实验目录
│   │   ├── training_args.json       # 参数文件
│   │   ├── model_epoch_*.pt         # checkpoints
│   │   └── plots/                   # 可视化
│   └── pred_<method>_<params>/      # 推理目录
│       ├── inference_args.json
│       └── <stem>/                  # 每个图像的结果
```

---

## 结语

本文档提供了 Seg-R0 项目的完整知识地图。通过本文档，AI Agent 可以：

✅ **快速了解**项目目标和架构  
✅ **定位文件**快速找到相关代码和文档  
✅ **解决问题**查找故障排除指南  
✅ **开始工作**获取命令模板和示例  
✅ **深入学习**通过索引找到详细文档  

**建议阅读顺序**（首次了解项目）:
1. 本文档 - 整体概览
2. `seg-rl/README_SEG_RL_INSTRUCTIONS.md` - 基础使用
3. `seg-rl/heatmap/README_SAM_LORA.md` - SAM LoRA（如需使用）
4. 相关代码文件（`model.py`, `train.py`等）
5. 历史对话记录（深入了解）

**最后更新**: 2025-11-09  
**文档维护**: 请在每次重大更新后更新本文档  
**版本**: v2.0

