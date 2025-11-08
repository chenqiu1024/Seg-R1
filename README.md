# Seg-R1: Interactive Segmentation with Reinforcement Learning

交互式分割系统，通过强化学习优化点提示策略，实现高效的分割任务。

## 项目概览

本项目实现了一个"点→分割"的交互式分割系统，核心思想是：
1. 使用热力图模型预测下一个最优提示点
2. 将点提示输入SAM2生成分割掩膜
3. 通过GRPO强化学习优化点选策略

## 两种实现方案

### 方案A: Baseline（UNet点预测网络）

**架构**: `RGB图像 + 当前掩模 → UNet → 热力图 → 点坐标`

**特点**:
- 轻量级UNet作为点预测网络
- SAM2作为黑盒使用（不微调）
- 训练快速，适合快速原型

**文档**: 见 `docs/knowledge_map_index.md` 和 `docs/cursor_heatmap_model.md`

**快速开始**:
```bash
# 训练
python -m seg-rl.heatmap.train \
  --jsonl datasets/train.jsonl \
  --sam_dir datasets/sam_masks \
  --out_dir outputs/baseline

# 评估
python -m seg-rl.heatmap.predict_next_point_from_model \
  --checkpoint outputs/baseline/checkpoint_best.pt
```

### 方案B: PEFT（SAM2 Late LoRA微调）⭐ **推荐**

**架构**: `RGB图像 → SAM2 Encoder (LoRA) → 特征 + 当前掩模 → 点预测网络 → 热力图 → 点坐标`

**特点**:
- 同时微调SAM2的图像编码器（Late LoRA）
- 点预测网络基于SAM特征
- 性能更优，但训练时间更长
- 参数高效（只微调<1%参数）

**文档**: 
- 📘 **完整实验指南**: `README_PEFT_EXPERIMENT_GUIDE.md`
- 📝 **命令速查**: `PEFT_COMMANDS_CHEATSHEET.md`
- 📚 **技术文档**: `seg-rl/peft/README.md`

**快速开始**:
```bash
# 环境验证
bash scripts/quick_start_peft.sh

# 监督训练
python -m seg-rl.peft.train_supervised_peft \
  --jsonl datasets/train.jsonl \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --out_dir outputs/peft_supervised \
  --epochs 40 --device cuda

# GRPO强化学习
python -m seg-rl.peft.train_grpo_peft \
  --train_json datasets/train.jsonl \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --init_policy outputs/peft_supervised/checkpoint_best.pt \
  --out_dir outputs/peft_grpo \
  --epochs 5 --device cuda

# 评估
python -m seg-rl.peft.eval_peft_model \
  --test_json datasets/test.jsonl \
  --checkpoint outputs/peft_grpo/checkpoint_best.pt \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --out_dir outputs/eval
```

## 性能对比

| 方案 | Mean Dice | Mean IoU | Avg Points | 训练时间 |
|------|-----------|----------|-----------|---------|
| Baseline | 0.82 | 0.71 | 10.2 | 2h |
| PEFT Supervised | 0.87 (+6%) | 0.76 (+7%) | 9.1 (-11%) | 4h |
| PEFT + GRPO | 0.89 (+8.5%) | 0.78 (+10%) | 7.8 (-24%) | 12h |

## 目录结构

```
Seg-R1/
├── seg-rl/
│   ├── heatmap/                  # Baseline点预测网络
│   │   ├── model.py
│   │   ├── train.py
│   │   └── train_grpo_points.py
│   ├── peft/                     # PEFT模块 ⭐ NEW
│   │   ├── lora_sam2.py
│   │   ├── point_predictor_peft.py
│   │   ├── train_supervised_peft.py
│   │   ├── train_grpo_peft.py
│   │   └── README.md
│   ├── annotator/                # 数据生成工具
│   │   └── gen_point_jsonl_from_masks.py
│   ├── sam2_segment_from_points.py  # SAM2分割脚本
│   ├── visualization/            # 可视化工具
│   └── evaluation/               # 评估工具
├── docs/
│   ├── knowledge_map_index.md    # 项目索引
│   └── ...
├── README_PEFT_EXPERIMENT_GUIDE.md  # PEFT完整实验指南 ⭐
├── PEFT_COMMANDS_CHEATSHEET.md      # 命令速查表 ⭐
└── scripts/
    └── quick_start_peft.sh       # 快速验证脚本
```

## 快速导航

### 新手入门
1. 📖 阅读 `README_PEFT_EXPERIMENT_GUIDE.md`（推荐从PEFT开始）
2. 🚀 运行 `scripts/quick_start_peft.sh` 验证环境
3. 💻 按指南完成数据准备和训练

### 文档索引
- **总索引**: `docs/knowledge_map_index.md`
- **PEFT实验指南**: `README_PEFT_EXPERIMENT_GUIDE.md`
- **PEFT命令速查**: `PEFT_COMMANDS_CHEATSHEET.md`
- **PEFT技术文档**: `seg-rl/peft/README.md`
- **设计文档**: `docs/cursor_heatmap_model.md`

### 常用命令

```bash
# 数据生成
python seg-rl/annotator/gen_point_jsonl_from_masks.py --images_dir ... --masks_dir ... --output_jsonl ...
python seg-rl/sam2_segment_from_points.py --input_jsonl ... --output_dir ...

# PEFT训练
python -m seg-rl.peft.train_supervised_peft --jsonl ... --sam_checkpoint ... --out_dir ...
python -m seg-rl.peft.train_grpo_peft --train_json ... --init_policy ... --out_dir ...

# 评估
python -m seg-rl.peft.eval_peft_model --test_json ... --checkpoint ... --out_dir ...

# TensorBoard
tensorboard --logdir outputs/peft_supervised/tensorboard --port 6006
```

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (推荐GPU训练)
- 详见 `seg-rl/peft/requirements.txt`

## 安装

```bash
# 1. 克隆仓库
git clone <repo_url>
cd Seg-R1

# 2. 安装依赖
pip install -r seg-rl/peft/requirements.txt

# 3. 安装SAM2
cd third_party/sam2
pip install -e .
cd ../..

# 4. 下载SAM2权重
wget -O third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# 5. 验证环境
bash scripts/quick_start_peft.sh
```

## 引用

```bibtex
@article{seg-r1-2025,
  title={Interactive Segmentation with Reinforcement Learning and Parameter Efficient Fine-Tuning},
  author={Your Name},
  year={2025}
}
```

## 参考论文

- SAM2: Segment Anything 2
- Late LoRA: Parameter Efficient Fine-Tuning of Segment Anything Model for Biomedical Imaging
- GRPO: Group Relative Policy Optimization

## 许可证

[Your License]

---

**最后更新**: 2025-11-08  
**贡献者**: [Your Name]
