# PEFT (Parameter Efficient Fine-Tuning) for SAM2

本模块实现了对SAM2的参数高效微调，使用Late LoRA策略只在SAM2图像编码器的最后一个Transformer块注入LoRA适配器，同时训练一个基于SAM特征的点预测网络。

## 架构概述

### 原架构 vs 新架构

**原架构**:
```
RGB图像 + 当前掩模 → UNet → 热力图 → 点坐标
```

**新架构 (PEFT)**:
```
RGB图像 → SAM2 Encoder (Late LoRA) → 特征
                                      ↓ + 当前掩模
                              点预测网络 → 热力图 → 点坐标
```

### 训练流程

1. **监督预训练**: 使用启发式生成的点序列数据训练
2. **GRPO强化学习**: 通过与SAM2环境交互优化策略

## 模块说明

### 核心模块

- `lora_sam2.py`: LoRA注入与SAM2封装
- `point_predictor_peft.py`: 点预测网络（接受SAM特征作为输入）
- `datasets_peft.py`: 数据加载器
- `utils_peft.py`: 辅助工具函数
- `train_supervised_peft.py`: 监督预训练脚本
- `train_grpo_peft.py`: GRPO强化学习脚本
- `eval_peft_model.py`: 评估脚本

### 可视化模块

- `visualization/viz_sam_features.py`: SAM特征可视化
- `visualization/viz_peft_predictions.py`: 预测结果可视化

## 使用指南

### 1. 数据准备

使用现有的数据生成脚本生成训练数据（JSONL格式）：

```bash
# 生成训练数据（详见seg-rl/annotator/gen_point_jsonl_from_masks.py）
python seg-rl/annotator/gen_point_jsonl_from_masks.py \
  --images_dir datasets/Task01_BrainTumour/canonical/images \
  --masks_dir datasets/Task01_BrainTumour/canonical/masks \
  --output_jsonl datasets/Task01_BrainTumour/peft_train.jsonl
```

### 2. 监督预训练

```bash
python -m seg-rl.peft.train_supervised_peft \
  --jsonl datasets/Task01_BrainTumour/peft_train.jsonl \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --lora_rank 16 --lora_alpha 32 \
  --feature_scale 8 --fusion_mode film \
  --image_size 512 512 \
  --loss kl --sigma 8.0 --label_loss_weight 0.1 \
  --batch_size 8 --epochs 40 --amp \
  --lr_sam 1e-5 --lr_point 1e-4 \
  --out_dir outputs/braintumour/peft_supervised \
  --save_every 5 --auto_resume --tb
```

#### 关键参数说明

- `--lora_rank`: LoRA秩，控制适配器的表达能力（推荐: 8-32）
- `--lora_alpha`: LoRA缩放因子（推荐: rank的2倍）
- `--feature_scale`: SAM特征尺度，8表示H/8×W/8（推荐: 8）
- `--fusion_mode`: 特征融合模式，"film"或"concat"（推荐: film）
- `--freeze_sam`: 冻结SAM，只训练点预测网络（调试时使用）

### 3. GRPO强化学习

```bash
python -m seg-rl.peft.train_grpo_peft \
  --train_json datasets/Task01_BrainTumour/peft_train.jsonl \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --init_policy outputs/braintumour/peft_supervised/checkpoint_best.pt \
  --out_dir outputs/braintumour/peft_grpo \
  --height 512 --width 512 --stride 8 \
  --max_points 16 --group_size 4 \
  --epochs 5 --batch_size 8 \
  --lr_sam 5e-6 --lr_point 5e-5 \
  --beta_kl 0.02 --beta_entropy 0.01 \
  --clip_epsilon 0.2 \
  --tb --device cuda
```

#### GRPO参数说明

- `--group_size`: 计算相对优势的组大小（推荐: 4）
- `--clip_epsilon`: PPO裁剪系数（推荐: 0.1-0.3）
- `--beta_kl`: KL惩罚系数（推荐: 0.01-0.05）
- `--beta_entropy`: 熵正则系数（推荐: 0.005-0.02）
- `--pixel_temp_start/end`: 像素采样温度退火（从高到低，鼓励探索→利用）

### 4. 评估

```bash
python -m seg-rl.peft.eval_peft_model \
  --test_json datasets/Task01_BrainTumour/test.jsonl \
  --checkpoint outputs/braintumour/peft_grpo/checkpoint_best.pt \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --out_dir outputs/braintumour/peft_eval \
  --max_rollout_steps 16 \
  --device cuda
```

### 5. 可视化

#### SAM特征可视化

```bash
python -m seg-rl.visualization.viz_sam_features \
  --image path/to/image.jpg \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --lora_checkpoint outputs/braintumour/peft_supervised/checkpoint_best.pt \
  --out_dir outputs/viz/sam_features
```

#### 预测结果可视化

```bash
python -m seg-rl.visualization.viz_peft_predictions \
  --image path/to/image.jpg \
  --prev_mask path/to/prev_mask.png \
  --checkpoint outputs/braintumour/peft_supervised/checkpoint_best.pt \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --out_dir outputs/viz/predictions
```

## 实验建议

### 渐进式调试

**阶段1**: 冻结SAM，只训练点预测网络

```bash
python -m seg-rl.peft.train_supervised_peft \
  --jsonl ... \
  --freeze_sam \
  --lr_point 1e-4 \
  --epochs 20
```

- 验证数据加载正确
- 验证点预测网络架构合理
- 找到好的学习率和损失权重

**阶段2**: 解冻LoRA，联合训练

```bash
python -m seg-rl.peft.train_supervised_peft \
  --jsonl ... \
  --lr_sam 1e-5 --lr_point 1e-4 \
  --epochs 40
```

- 使用warmup（默认3 epochs）
- 监控LoRA权重范数
- 对比冻结vs解冻的性能提升

**阶段3**: GRPO微调

```bash
python -m seg-rl.peft.train_grpo_peft \
  --init_policy <best_supervised_ckpt> \
  --epochs 5
```

- 先用小batch调试rollout逻辑
- 检查奖励信号是否合理
- 逐步增加batch size和episode长度

### 超参数搜索空间

**LoRA**:
- rank: [4, 8, 16, 32]
- alpha: [rank, 2*rank]

**融合方式**:
- "film", "concat"

**学习率**:
- SAM: [5e-6, 1e-5, 2e-5]
- 点网络: [5e-5, 1e-4, 2e-4]

**GRPO**:
- beta_kl: [0.01, 0.02, 0.05]
- beta_entropy: [0.005, 0.01, 0.02]
- clip_epsilon: [0.1, 0.2, 0.3]

## Checkpoint结构

```python
checkpoint = {
    'epoch': int,
    'step': int,
    
    # 模型状态
    'point_predictor_state': OrderedDict,
    'sam_lora_state': OrderedDict,  # 只保存LoRA参数
    
    # 优化器状态
    'optimizer_sam_state': dict,
    'optimizer_point_state': dict,
    
    # 配置
    'config': {
        'lora_rank': int,
        'lora_alpha': int,
        'feature_scale': int,
        'fusion_mode': str,
        ...
    },
    
    # 指标
    'metrics': {...}
}
```

## 性能基准

与baseline（不用PEFT）对比：
- Dice提升: 预期+2-5%
- 点数减少: 预期-10-20%
- 训练时间: LoRA微调比全微调快2-3x

## 故障排除

### 常见问题

1. **CUDA Out of Memory**
   - 减小batch_size
   - 使用--amp开启混合精度
   - 减小image_size

2. **训练不收敛**
   - 检查学习率是否过大
   - 确认数据加载正确（k=0的掩模是否为全零）
   - 尝试冻结SAM先训练点网络

3. **GRPO奖励信号异常**
   - 检查SAM2掩模生成是否正常
   - 调整temperature和KL系数
   - 减小max_points避免过长rollout

## 参考文献

- Late LoRA方法: Parameter Efficient Fine-Tuning of Segment Anything Model for Biomedical Imaging
- GRPO算法: Group Relative Policy Optimization

## 许可证

本模块遵循项目整体许可证。

