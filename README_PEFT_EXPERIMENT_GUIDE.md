# PEFT + SAM2 实验指导手册

本文档提供完整的实验流程指导，从环境准备到模型训练、评估和分析。

## 目录

- [环境准备](#环境准备)
- [数据准备](#数据准备)
- [实验流程](#实验流程)
  - [阶段0: 验证环境](#阶段0-验证环境)
  - [阶段1: 生成训练数据](#阶段1-生成训练数据)
  - [阶段2: 监督预训练](#阶段2-监督预训练)
  - [阶段3: GRPO强化学习](#阶段3-grpo强化学习)
  - [阶段4: 评估与分析](#阶段4-评估与分析)
- [高级用法](#高级用法)
- [故障排除](#故障排除)

---

## 环境准备

### 1. Python环境

确保已安装Python 3.8+和必要的依赖：

```bash
# 创建虚拟环境（推荐）
conda create -n seg-r1 python=3.10
conda activate seg-r1

# 安装PyTorch（根据你的CUDA版本）
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 或 CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install numpy scipy pillow opencv-python tqdm tensorboard scikit-learn
```

### 2. SAM2安装

```bash
# 克隆SAM2到third_party目录（如果还没有）
cd third_party
git clone https://github.com/facebookresearch/segment-anything-2.git sam2
cd sam2
pip install -e .
cd ../..
```

### 3. 下载SAM2模型权重

```bash
# 创建checkpoint目录
mkdir -p third_party/sam2/checkpoints

# 下载SAM2.1 Hiera Large模型 (~900MB)
wget -O third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# 或者使用curl
curl -L -o third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

---

## 数据准备

### 数据格式要求

您需要准备以下数据：

```
datasets/
└── Task01_BrainTumour/           # 数据集名称
    └── canonical/
        ├── images/                # 原始图像
        │   ├── sample_001.jpg
        │   ├── sample_002.jpg
        │   └── ...
        └── masks/                 # 真值掩模（PNG，单通道）
            ├── sample_001.png
            ├── sample_002.png
            └── ...
```

**要求**：
- 图像和掩模文件名（stem）必须一一对应
- 掩模为单通道PNG，前景>127，背景≤127
- 图像格式：JPG, JPEG, PNG

### 示例数据集

如果使用Brain Tumour数据集：

```bash
# 假设你已经下载了数据集
# 确保目录结构符合上述格式
ls datasets/Task01_BrainTumour/canonical/images/ | head -5
ls datasets/Task01_BrainTumour/canonical/masks/ | head -5
```

---

## 实验流程

### 阶段0: 验证环境

在开始训练前，先验证所有模块是否正常工作：

```bash
# 运行基础功能测试
python -m seg-rl.peft.test_modules --device cuda

# 预期输出：
# ================================================================================
# PEFT Modules Testing
# ================================================================================
# Device: cuda
# ...
# ✓ All tests passed!
```

如果测试失败，请检查依赖安装和CUDA配置。

---

### 阶段1: 生成训练数据

使用启发式算法生成点序列训练数据。这是一个**交替执行**的过程。

#### 步骤1.1: 生成第一个提示点

```bash
python seg-rl/annotator/gen_point_jsonl_from_masks.py \
  --images_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/images \
  --masks_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/masks \
  --output_jsonl outputs/braintumour/pretrain_251107.jsonl
```

**输出**：`pretrain_251107.jsonl`，每个样本包含第一个点：
```json
{
  "image": "/abs/path/to/sample_001.jpg",
  "gt_mask": "/abs/path/to/sample_001.png",
  "points": [[x0, y0]],
  "labels": [1]
}
```

#### 步骤1.2: 用SAM2生成第一个掩模

```bash
python seg-rl/sam2_segment_from_points.py \
  --input_jsonl outputs/braintumour/pretrain_251107.jsonl \
  --json_output outputs/braintumour/pretrain_251107.jsonl \
  --output_dir outputs/braintumour/sam_masks_ref \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --device cuda \
  --resize 512 512 \
  --skip_existing
```

**输出**：
- `sam_masks_ref/sample_001/0.png` - 第一个点生成的掩模
- 更新`pretrain_251107.jsonl`，添加`sam_masks_dir`字段

#### 步骤1.3: 生成第二个提示点

```bash
python seg-rl/annotator/gen_point_jsonl_from_masks.py \
  --appendto_jsonl outputs/braintumour/pretrain_251107.jsonl 
```

**输出**：更新JSONL，添加第二个点：
```json
{
  "points": [[x0, y0], [x1, y1]],
  "labels": [1, 1],
  "sam_masks_dir": "..."
}
```

#### 步骤1.4: 生成第二个掩模

```bash
python seg-rl/sam2_segment_from_points.py \
  --input_jsonl outputs/braintumour/pretrain_251107.jsonl \
  --json_output outputs/braintumour/pretrain_251107.jsonl \
  --output_dir outputs/braintumour/sam_masks_ref \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --device cuda \
  --resize 512 512 \
  --skip_existing
```

**输出**：`sam_masks_ref/sample_001/1.png`

#### 步骤1.5-1.N: 继续迭代

重复步骤1.3和1.4，直到达到期望的点数（通常8-16个点）：

```bash
# 一键脚本（循环N次）
for i in {2..15}; do
  echo "=== Generating point $i ==="
  python seg-rl/annotator/gen_point_jsonl_from_masks.py \
    --appendto_jsonl outputs/braintumour/pretrain_251107.jsonl 
  
  python seg-rl/sam2_segment_from_points.py \
    --input_jsonl outputs/braintumour/pretrain_251107.jsonl \
    --json_output outputs/braintumour/pretrain_251107.jsonl \
    --output_dir outputs/braintumour/sam_masks_ref \
    --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
    --device cuda --resize 512 512 --skip_existing
done
```

#### 步骤1.6: 验证数据

```bash
# 检查生成的数据
python -c "
import json
with open('outputs/braintumour/pretrain_251107.jsonl', 'r') as f:
    data = json.load(f)
    print(f'Total samples: {len(data)}')
    print(f'Points per sample: {len(data[0][\"points\"])}')
    print(f'Sample entry: {data[0].keys()}')
"

# 检查掩模文件
ls outputs/braintumour/sam_masks_ref/BRATS_001_z0029/ | wc -l
# 应该输出：16（如果生成了16个点）
```

#### 步骤1.7: 划分训练/测试集（可选）

```bash
# 使用Python脚本划分（80% train, 20% test）
python -c "
import json
import random

with open('outputs/braintumour/pretrain_251107.jsonl', 'r') as f:
    data = json.load(f)

random.seed(42)
random.shuffle(data)

split_idx = int(len(data) * 0.8)
train_data = data[:split_idx]
test_data = data[split_idx:]

with open('outputs/braintumour/peft_train-251107.jsonl', 'w') as f:
    json.dump(train_data, f, indent=2)

with open('outputs/braintumour/peft_test-251107.jsonl', 'w') as f:
    json.dump(test_data, f, indent=2)

print(f'Train: {len(train_data)}, Test: {len(test_data)}')
"
```

---

### 阶段2: 监督预训练

使用启发式生成的数据训练SAM2 LoRA + 点预测网络。

#### 实验2.1: 基础训练（推荐配置）

```bash
python -m seg-rl.peft.train_supervised_peft \
  --jsonl outputs/braintumour/peft_train-251107.jsonl \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --lora_rank 16 \
  --lora_alpha 32 \
  --feature_scale 8 \
  --fusion_mode film \
  --image_size 512 512 \
  --loss kl \
  --sigma 8.0 \
  --tau 1.0 \
  --label_loss_weight 0.1 \
  --batch_size 8 \
  --epochs 40 \
  --amp \
  --lr_sam 1e-5 \
  --lr_point 1e-4 \
  --weight_decay 1e-4 \
  --grad_clip 1.0 \
  --lr_scheduler warmup_cosine \
  --warmup_epochs 3 \
  --val_ratio 0.1 \
  --eval_thresholds "8,12,16,20" \
  --out_dir outputs/braintumour/peft_supervised_baseline-251107 \
  --save_every 5 \
  --auto_resume \
  --tb \
  --device cuda \
  --num_workers 4 \
  --seed 42
```

**参数说明**：
- `--lora_rank 16`: LoRA秩，控制适配器容量（典型值：8-32）
- `--lora_alpha 32`: LoRA缩放，通常是rank的2倍
- `--fusion_mode film`: 特征融合方式（film=FiLM条件化，concat=拼接）
- `--loss kl`: 热力图损失类型（kl=KL散度，mse=均方误差）
- `--sigma 8.0`: 高斯目标的标准差（像素）
- `--amp`: 启用混合精度，节省显存
- `--lr_sam 1e-5`: SAM LoRA学习率（小于点网络）
- `--lr_point 1e-4`: 点预测网络学习率
- `--tb`: 启用TensorBoard日志

**预期运行时间**（单个RTX 3090）：
- ~3-4小时（40 epochs，1000张图像）

**监控训练**：

```bash
# 启动TensorBoard
tensorboard --logdir outputs/braintumour/peft_supervised_baseline-251107/tensorboard --port 6006

# 在浏览器打开 http://localhost:6006
# 查看：
# - train/total_loss, train/heatmap_loss, train/label_loss
# - val/val_loss, val/val_pck@8, val/val_pck@12, ...
```

**查看输出**：

```bash
# 训练日志
tail -f outputs/braintumour/peft_supervised_baseline-251107/train.log

# Checkpoint列表
ls outputs/braintumour/peft_supervised_baseline/*.pt
# checkpoint_epoch005.pt, checkpoint_epoch010.pt, ..., checkpoint_best.pt
```

#### 实验2.2: 冻结SAM（调试用）

先验证点预测网络是否能学习，冻结SAM只训练点网络：

```bash
python -m seg-rl.peft.train_supervised_peft \
  --jsonl datasets/Task01_BrainTumour/peft_train.jsonl \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --freeze_sam \
  --lora_rank 16 \
  --lora_alpha 32 \
  --feature_scale 8 \
  --fusion_mode film \
  --image_size 512 512 \
  --loss kl \
  --sigma 8.0 \
  --batch_size 8 \
  --epochs 20 \
  --lr_point 1e-4 \
  --out_dir outputs/braintumour/peft_frozen_sam \
  --save_every 5 \
  --tb \
  --device cuda
```

**用途**：
- 快速验证数据加载正确
- 找到点网络的合适学习率
- 作为baseline对比

#### 实验2.3: 超参数搜索

探索不同配置的性能：

```bash
# LoRA秩对比
for rank in 8 16 32; do
  python -m seg-rl.peft.train_supervised_peft \
    --jsonl datasets/Task01_BrainTumour/peft_train.jsonl \
    --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
    --lora_rank $rank \
    --lora_alpha $((rank * 2)) \
    --out_dir outputs/braintumour/peft_rank${rank} \
    --epochs 40 --batch_size 8 --tb --device cuda \
    # ... 其他参数同上
done

# 融合方式对比
for fusion in film concat; do
  python -m seg-rl.peft.train_supervised_peft \
    --jsonl datasets/Task01_BrainTumour/peft_train.jsonl \
    --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
    --fusion_mode $fusion \
    --out_dir outputs/braintumour/peft_fusion_${fusion} \
    --epochs 40 --batch_size 8 --tb --device cuda \
    # ... 其他参数同上
done
```

---

### 阶段3: GRPO强化学习

在监督预训练的基础上，使用GRPO进一步优化策略。

#### 实验3.1: 基础GRPO训练

```bash
python -m seg-rl.peft.train_grpo_peft \
  --train_json datasets/Task01_BrainTumour/peft_train.jsonl \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --init_policy outputs/braintumour/peft_supervised_baseline/checkpoint_best.pt \
  --out_dir outputs/braintumour/peft_grpo_baseline \
  --height 512 \
  --width 512 \
  --stride 8 \
  --max_points 16 \
  --group_size 4 \
  --epochs 5 \
  --batch_size 8 \
  --lr_sam 5e-6 \
  --lr_point 5e-5 \
  --grad_clip 1.0 \
  --clip_epsilon 0.2 \
  --beta_kl 0.02 \
  --beta_entropy 0.01 \
  --pixel_temp_start 1.5 \
  --pixel_temp_end 0.8 \
  --label_temp_start 1.2 \
  --label_temp_end 0.8 \
  --dice_stop_threshold 0.95 \
  --improvement_stop_threshold 0.001 \
  --save_every 100 \
  --tb \
  --device cuda \
  --seed 42
```

**GRPO参数说明**：
- `--init_policy`: 监督预训练的最佳模型作为初始化
- `--group_size 4`: GRPO组大小，计算相对优势
- `--clip_epsilon 0.2`: PPO裁剪系数
- `--beta_kl 0.02`: KL惩罚系数（防止偏离参考策略太远）
- `--beta_entropy 0.01`: 熵正则系数（鼓励探索）
- `--pixel_temp_start/end`: 温度退火（从探索到利用）

**预期运行时间**（单个RTX 3090）：
- ~6-8小时（5 epochs，1000张图像，batch_size=8）
- 每个样本需要执行rollout，比监督训练慢

**监控GRPO训练**：

```bash
# TensorBoard
tensorboard --logdir outputs/braintumour/peft_grpo_baseline/tensorboard --port 6007

# 查看：
# - train/mean_reward: 每个batch的平均奖励
# - train/policy_loss, train/kl_loss, train/entropy
# - train/pixel_temp: 温度退火曲线
```

#### 实验3.2: 调整GRPO超参数

```bash
# 增大KL惩罚（更保守，不容易崩溃）
python -m seg-rl.peft.train_grpo_peft \
  --init_policy outputs/braintumour/peft_supervised_baseline/checkpoint_best.pt \
  --beta_kl 0.05 \
  --out_dir outputs/braintumour/peft_grpo_kl005 \
  # ... 其他参数同上

# 减小KL惩罚（更激进，可能收益更大但不稳定）
python -m seg-rl.peft.train_grpo_peft \
  --init_policy outputs/braintumour/peft_supervised_baseline/checkpoint_best.pt \
  --beta_kl 0.01 \
  --out_dir outputs/braintumour/peft_grpo_kl001 \
  # ... 其他参数同上
```

---

### 阶段4: 评估与分析

#### 实验4.1: 评估监督预训练模型

```bash
python -m seg-rl.peft.eval_peft_model \
  --test_json datasets/Task01_BrainTumour/peft_test.jsonl \
  --checkpoint outputs/braintumour/peft_supervised_baseline/checkpoint_best.pt \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --out_dir outputs/braintumour/eval_supervised \
  --image_size 512 512 \
  --max_rollout_steps 16 \
  --dice_threshold 0.95 \
  --improvement_threshold 0.001 \
  --device cuda \
  --save_visualizations
```

**输出**：
```
outputs/braintumour/eval_supervised/
├── eval_results.json          # 详细结果
└── visualizations/            # 可视化图像（如果启用）
    ├── sample_001_rollout.png
    └── ...
```

**查看结果**：

```bash
# 查看汇总统计
cat outputs/braintumour/eval_supervised/eval_results.json | jq '.summary'

# 输出示例：
# {
#   "num_samples": 200,
#   "mean_dice": 0.8523,
#   "std_dice": 0.1234,
#   "median_dice": 0.8756,
#   "mean_iou": 0.7654,
#   "mean_num_points": 8.3
# }
```

#### 实验4.2: 评估GRPO模型

```bash
python -m seg-rl.peft.eval_peft_model \
  --test_json datasets/Task01_BrainTumour/peft_test.jsonl \
  --checkpoint outputs/braintumour/peft_grpo_baseline/checkpoint_best.pt \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --out_dir outputs/braintumour/eval_grpo \
  --image_size 512 512 \
  --max_rollout_steps 16 \
  --device cuda
```

#### 实验4.3: 对比baseline（无PEFT）

如果你有之前训练的baseline模型：

```bash
# 评估原始UNet模型（无PEFT）
python -m seg-rl.heatmap.predict_next_point_from_model \
  --checkpoint outputs/braintumour/baseline_unet/checkpoint_best.pt \
  --test_json datasets/Task01_BrainTumour/peft_test.jsonl \
  --out_dir outputs/braintumour/eval_baseline \
  # ... 其他参数
```

**性能对比表格**：

```bash
# 创建对比报告
python -c "
import json

# 加载结果
with open('outputs/braintumour/eval_supervised/eval_results.json', 'r') as f:
    supervised = json.load(f)['summary']

with open('outputs/braintumour/eval_grpo/eval_results.json', 'r') as f:
    grpo = json.load(f)['summary']

# 打印对比
print('Model Comparison:')
print(f'{"Metric":<20} {"Supervised":<15} {"GRPO":<15} {"Improvement":<15}')
print('-' * 65)

for key in ['mean_dice', 'mean_iou', 'mean_num_points']:
    sup_val = supervised[key]
    grpo_val = grpo[key]
    if 'num_points' in key:
        # 点数越少越好
        improv = (sup_val - grpo_val) / sup_val * 100
        print(f'{key:<20} {sup_val:<15.4f} {grpo_val:<15.4f} {improv:+.2f}%')
    else:
        # Dice/IoU越大越好
        improv = (grpo_val - sup_val) / sup_val * 100
        print(f'{key:<20} {sup_val:<15.4f} {grpo_val:<15.4f} {improv:+.2f}%')
"
```

#### 实验4.4: 可视化分析

**SAM特征对比（LoRA微调前后）**：

```bash
# 选择一张测试图像
python -m seg-rl.visualization.viz_sam_features \
  --image datasets/Task01_BrainTumour/canonical/images/sample_001.jpg \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --lora_checkpoint outputs/braintumour/peft_supervised_baseline/checkpoint_best.pt \
  --out_dir outputs/braintumour/viz_sam_features \
  --feature_scale 8 \
  --device cuda
```

**输出**：
```
outputs/braintumour/viz_sam_features/
├── sam_features_original.png   # 原始SAM特征（PCA降维）
├── sam_features_lora.png       # LoRA微调后特征
└── sam_features_diff.png       # 差异图
```

**预测热力图可视化**：

```bash
# 可视化模型预测
python -m seg-rl.visualization.viz_peft_predictions \
  --image datasets/Task01_BrainTumour/canonical/images/sample_001.jpg \
  --prev_mask datasets/Task01_BrainTumour/sam_masks_peft/sample_001/0.png \
  --checkpoint outputs/braintumour/peft_supervised_baseline/checkpoint_best.pt \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --out_dir outputs/braintumour/viz_predictions/sample_001_step1 \
  --device cuda
```

**输出**：
```
outputs/braintumour/viz_predictions/sample_001_step1/
├── heatmap.png      # 预测热力图
├── overlay.png      # 热力图+点叠加在原图
└── prev_mask.png    # 前一步掩模
```

---

## 高级用法

### 多GPU训练

```bash
# 使用DataParallel（简单，但效率较低）
python -m seg-rl.peft.train_supervised_peft \
  --device cuda \
  --batch_size 16 \
  # ... 其他参数

# 使用DistributedDataParallel（推荐，效率高）
# 需要修改代码添加DDP支持
```

### 断点续传

```bash
# 自动续传（从最新checkpoint）
python -m seg-rl.peft.train_supervised_peft \
  --auto_resume \
  --out_dir outputs/braintumour/peft_supervised_baseline \
  # ... 其他参数

# 从指定checkpoint续传
python -m seg-rl.peft.train_supervised_peft \
  --resume outputs/braintumour/peft_supervised_baseline/checkpoint_epoch020.pt \
  --out_dir outputs/braintumour/peft_supervised_baseline \
  # ... 其他参数
```

### 调整图像分辨率

根据GPU显存调整：

```bash
# 小显存（8GB）- 使用256x256
python -m seg-rl.peft.train_supervised_peft \
  --image_size 256 256 \
  --batch_size 16 \
  --amp \
  # ...

# 中等显存（16GB）- 使用512x512（推荐）
python -m seg-rl.peft.train_supervised_peft \
  --image_size 512 512 \
  --batch_size 8 \
  --amp \
  # ...

# 大显存（24GB+）- 使用768x768
python -m seg-rl.peft.train_supervised_peft \
  --image_size 768 768 \
  --batch_size 4 \
  --amp \
  # ...
```

### 使用不同SAM2模型

```bash
# SAM2.1 Base+ (更快，略低精度)
python -m seg-rl.peft.train_supervised_peft \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_base_plus.pt \
  # ...

# SAM2.1 Small (最快，适合快速实验)
python -m seg-rl.peft.train_supervised_peft \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_small.pt \
  # ...
```

---

## 故障排除

### 问题1: CUDA Out of Memory

**症状**：
```
RuntimeError: CUDA out of memory. Tried to allocate XXX MiB
```

**解决方案**：
```bash
# 1. 减小batch size
--batch_size 4  # 或更小

# 2. 启用混合精度
--amp

# 3. 减小图像尺寸
--image_size 256 256

# 4. 减小LoRA rank
--lora_rank 8

# 5. 清理GPU缓存
python -c "import torch; torch.cuda.empty_cache()"
```

### 问题2: 数据加载错误

**症状**：
```
FileNotFoundError: sam_masks_dir/.../0.png not found
```

**解决方案**：
```bash
# 检查JSONL文件中的sam_masks_dir字段
python -c "
import json
with open('datasets/Task01_BrainTumour/peft_train.jsonl', 'r') as f:
    data = json.load(f)
    print('sam_masks_dir:', data[0].get('sam_masks_dir'))
"

# 确保掩模文件存在
ls datasets/Task01_BrainTumour/sam_masks_peft/sample_001/

# 如果缺失，重新运行sam2_segment_from_points.py
```

### 问题3: LoRA注入失败

**症状**：
```
RuntimeError: Cannot find qkv or q/k/v projection layers in attention
```

**解决方案**：

这可能是因为SAM2版本更新导致模型结构变化。检查`lora_sam2.py`中的`_inject_lora`方法，确保与你的SAM2版本兼容。

### 问题4: 训练loss不下降

**可能原因**：

1. **学习率过大或过小**
   ```bash
   # 尝试调整学习率
   --lr_sam 5e-6  # 减小SAM学习率
   --lr_point 5e-5  # 减小点网络学习率
   ```

2. **数据问题**
   ```bash
   # 检查数据分布
   python -c "
   import json
   import numpy as np
   with open('datasets/Task01_BrainTumour/peft_train.jsonl', 'r') as f:
       data = json.load(f)
       num_points = [len(d['points']) for d in data]
       print(f'Points per sample: mean={np.mean(num_points):.1f}, min={min(num_points)}, max={max(num_points)}')
   "
   ```

3. **先冻结SAM训练点网络**
   ```bash
   python -m seg-rl.peft.train_supervised_peft --freeze_sam ...
   ```

### 问题5: GRPO训练reward异常

**症状**：reward一直为负或不增长

**解决方案**：

1. **检查初始策略质量**
   ```bash
   # 确保监督预训练模型性能足够好
   # Dice应该 > 0.7
   ```

2. **调整KL系数**
   ```bash
   # 增大beta_kl，防止策略崩溃
   --beta_kl 0.05
   ```

3. **检查SAM2掩模生成**
   ```bash
   # 手动测试SAM2
   python -m seg-rl.sam2_segment_from_points \
     --input_jsonl test_single_sample.jsonl \
     --output_dir test_sam_output \
     --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt
   ```

### 问题6: ModuleNotFoundError: No module named 'iopath'

**症状**：
```
ModuleNotFoundError: No module named 'iopath'
ImportError: Error loading 'sam2.modeling.backbones.hieradet.Hiera':
ModuleNotFoundError("No module named 'iopath'")
```

**原因**：
`iopath` 是 SAM2 的必需依赖，但在安装 SAM2 时可能没有正确安装。

**解决方案**：
```bash
# 方法1: 直接安装 iopath
pip install iopath

# 方法2: 重新安装 SAM2 并确保依赖被安装
cd third_party/sam2
pip install -e . --force-reinstall
cd ../..

# 验证安装
python -c "from iopath.common.file_io import g_pathmgr; print('iopath 安装成功')"
```

### 问题7: RuntimeError: Cannot find stages in Hiera trunk

**症状**：
```
RuntimeError: Cannot find stages in Hiera trunk
```

**原因**：
代码尝试访问 Hiera trunk 的 `stages` 属性，但 Hiera 模型实际使用的是 `blocks` 属性（`nn.ModuleList`）来存储所有块。

**解决方案**：
这个问题已经在最新版本的代码中修复。如果仍然遇到此错误，请确保使用最新版本的 `lora_sam2.py`。修复后的代码会：
- 使用 `trunk.blocks` 而不是 `trunk.stages`
- 直接访问最后一个块：`trunk.blocks[-1]`

如果问题仍然存在，可以手动检查模型结构：
```bash
python -c "
import sys
sys.path.insert(0, 'third_party/sam2')
from sam2.build_sam import build_sam2
model = build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml', 
                   'third_party/sam2/checkpoints/sam2.1_hiera_large.pt', 
                   device='cpu')
trunk = model.image_encoder.trunk
print(f'Has blocks: {hasattr(trunk, \"blocks\")}')
print(f'Number of blocks: {len(trunk.blocks)}')
print(f'Last block has attn: {hasattr(trunk.blocks[-1], \"attn\")}')
"
```

### 问题8: RuntimeError: Expected all tensors to be on the same device

**症状**：
```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
(when checking argument for argument mat2 in method wrapper_CUDA_mm)
```

**原因**：
LoRA 参数（`lora_A` 和 `lora_B`）在创建时默认在 CPU 上，而模型和输入数据在 CUDA 上，导致设备不匹配。

**解决方案**：
这个问题已经在最新版本的代码中修复。修复后的代码会在创建 LoRA 参数时自动检测 `base_linear` 的设备，并将 LoRA 参数创建在同一设备上。

如果问题仍然存在，可以手动检查：
```bash
python -c "
import torch
import sys
sys.path.insert(0, 'third_party/sam2')
from sam2.build_sam import build_sam2
sys.path.insert(0, 'seg-rl/peft')
from lora_sam2 import LoRALinear

model = build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml', 
                   'third_party/sam2/checkpoints/sam2.1_hiera_large.pt', 
                   device='cuda')
trunk = model.image_encoder.trunk
last_block = trunk.blocks[-1]
attn = last_block.attn

print(f'Original qkv device: {attn.qkv.weight.device}')
lora_qkv = LoRALinear(attn.qkv, rank=16, alpha=32)
print(f'LoRA A device: {lora_qkv.lora_A.device}')
print(f'LoRA B device: {lora_qkv.lora_B.device}')
"
```

### 问题9: RuntimeError: The size of tensor a (8) must match the size of tensor b (64)

**症状**：
```
RuntimeError: The size of tensor a (8) must match the size of tensor b (64) at non-singleton dimension 2
```

**原因**：
SAM2 的 `_prepare_backbone_features` 方法返回的特征格式是 `[HW, B, C]`（展平格式），而 FiLM 融合层期望的格式是 `[B, C, H, W]`。如果直接使用返回的特征，会导致空间维度不匹配。

**解决方案**：
这个问题已经在最新版本的代码中修复。修复后的 `get_image_features` 方法会：
1. 获取 SAM2 返回的特征（格式：`[HW, B, C]`）
2. 获取对应的特征尺寸 `(H, W)`
3. 将特征从 `[HW, B, C]` 转换为 `[B, C, H, W]`

如果问题仍然存在，可以手动检查：
```bash
python -c "
import torch
import sys
sys.path.insert(0, 'third_party/sam2')
from sam2.build_sam import build_sam2
sys.path.insert(0, 'seg-rl/peft')
from lora_sam2 import LoRASAM2Wrapper
from point_predictor_peft import MaskEncoder

sam2_lora = LoRASAM2Wrapper(
    sam_checkpoint='third_party/sam2/checkpoints/sam2.1_hiera_large.pt',
    lora_rank=16,
    lora_alpha=32,
    device='cuda'
)

test_image = torch.randn(1, 3, 512, 512).to('cuda')
sam_features = sam2_lora.get_image_features(test_image, feature_scale=8)
print(f'SAM features shape: {sam_features.shape}')

test_mask = torch.randn(1, 1, 512, 512).to('cuda')
mask_encoder = MaskEncoder(output_channels=64, output_scale=8).to('cuda')
mask_features = mask_encoder(test_mask)
print(f'Mask features shape: {mask_features.shape}')

print(f'空间维度匹配: {sam_features.shape[2:] == mask_features.shape[2:]}')
"
```

---

## 完整实验流程示例

以下是一个完整的端到端实验脚本：

```bash
#!/bin/bash
# complete_peft_experiment.sh

set -e  # 遇到错误立即退出

# ============================================================================
# 配置
# ============================================================================
DATASET_NAME="Task01_BrainTumour"
DATASET_DIR="datasets/${DATASET_NAME}"
SAM_CHECKPOINT="third_party/sam2/checkpoints/sam2.1_hiera_large.pt"
OUTPUT_BASE="outputs/${DATASET_NAME}"
DEVICE="cuda"

# ============================================================================
# 阶段1: 数据准备
# ============================================================================
echo "=== Stage 1: Data Preparation ==="

# 生成第一个点
python seg-rl/annotator/gen_point_jsonl_from_masks.py \
  --images_dir ${DATASET_DIR}/canonical/images \
  --masks_dir ${DATASET_DIR}/canonical/masks \
  --output_jsonl ${DATASET_DIR}/peft_data.jsonl

# 迭代生成15个点
for i in {0..14}; do
  echo "Generating point $((i+1))/15..."
  
  # SAM2分割
  python seg-rl/sam2_segment_from_points.py \
    --input_jsonl ${DATASET_DIR}/peft_data.jsonl \
    --json_output ${DATASET_DIR}/peft_data.jsonl \
    --output_dir ${DATASET_DIR}/sam_masks_peft \
    --sam_checkpoint ${SAM_CHECKPOINT} \
    --device ${DEVICE} \
    --resize 512 512 \
    --skip_existing
  
  # 生成下一个点（最后一轮不需要）
  if [ $i -lt 14 ]; then
    python seg-rl/annotator/gen_point_jsonl_from_masks.py \
      --appendto_jsonl ${DATASET_DIR}/peft_data.jsonl
  fi
done

# 划分训练/测试集
python -c "
import json, random
with open('${DATASET_DIR}/peft_data.jsonl', 'r') as f:
    data = json.load(f)
random.seed(42)
random.shuffle(data)
split = int(len(data) * 0.8)
with open('${DATASET_DIR}/peft_train.jsonl', 'w') as f:
    json.dump(data[:split], f, indent=2)
with open('${DATASET_DIR}/peft_test.jsonl', 'w') as f:
    json.dump(data[split:], f, indent=2)
print(f'Train: {split}, Test: {len(data)-split}')
"

# ============================================================================
# 阶段2: 监督预训练
# ============================================================================
echo "=== Stage 2: Supervised Pretraining ==="

python -m seg-rl.peft.train_supervised_peft \
  --jsonl ${DATASET_DIR}/peft_train.jsonl \
  --sam_checkpoint ${SAM_CHECKPOINT} \
  --lora_rank 16 --lora_alpha 32 \
  --feature_scale 8 --fusion_mode film \
  --image_size 512 512 \
  --loss kl --sigma 8.0 --label_loss_weight 0.1 \
  --batch_size 8 --epochs 40 --amp \
  --lr_sam 1e-5 --lr_point 1e-4 \
  --lr_scheduler warmup_cosine --warmup_epochs 3 \
  --out_dir ${OUTPUT_BASE}/peft_supervised \
  --save_every 5 --auto_resume --tb \
  --device ${DEVICE} --seed 42

# ============================================================================
# 阶段3: GRPO强化学习
# ============================================================================
echo "=== Stage 3: GRPO Training ==="

python -m seg-rl.peft.train_grpo_peft \
  --train_json ${DATASET_DIR}/peft_train.jsonl \
  --sam_checkpoint ${SAM_CHECKPOINT} \
  --init_policy ${OUTPUT_BASE}/peft_supervised/checkpoint_best.pt \
  --out_dir ${OUTPUT_BASE}/peft_grpo \
  --height 512 --width 512 --stride 8 \
  --max_points 16 --group_size 4 \
  --epochs 5 --batch_size 8 \
  --lr_sam 5e-6 --lr_point 5e-5 \
  --beta_kl 0.02 --beta_entropy 0.01 \
  --clip_epsilon 0.2 \
  --save_every 100 --tb \
  --device ${DEVICE} --seed 42

# ============================================================================
# 阶段4: 评估
# ============================================================================
echo "=== Stage 4: Evaluation ==="

# 评估监督模型
python -m seg-rl.peft.eval_peft_model \
  --test_json ${DATASET_DIR}/peft_test.jsonl \
  --checkpoint ${OUTPUT_BASE}/peft_supervised/checkpoint_best.pt \
  --sam_checkpoint ${SAM_CHECKPOINT} \
  --out_dir ${OUTPUT_BASE}/eval_supervised \
  --device ${DEVICE}

# 评估GRPO模型
python -m seg-rl.peft.eval_peft_model \
  --test_json ${DATASET_DIR}/peft_test.jsonl \
  --checkpoint ${OUTPUT_BASE}/peft_grpo/checkpoint_best.pt \
  --sam_checkpoint ${SAM_CHECKPOINT} \
  --out_dir ${OUTPUT_BASE}/eval_grpo \
  --device ${DEVICE}

echo "=== Experiment Complete! ==="
echo "Results saved in: ${OUTPUT_BASE}"
```

**运行完整实验**：

```bash
chmod +x complete_peft_experiment.sh
./complete_peft_experiment.sh 2>&1 | tee experiment.log
```

---

## 预期结果

基于类似的医学图像分割任务，预期性能提升：

| 指标 | Baseline (UNet) | Supervised PEFT | GRPO PEFT | 改进 |
|------|----------------|-----------------|-----------|------|
| Mean Dice | 0.82 | 0.87 (+6%) | 0.89 (+8.5%) | ✓ |
| Mean IoU | 0.71 | 0.76 (+7%) | 0.78 (+10%) | ✓ |
| Avg Points | 10.2 | 9.1 (-11%) | 7.8 (-24%) | ✓ |
| Training Time | 2h | 4h | 12h | - |

---

## 引用

如果本工作对您的研究有帮助，请引用：

```bibtex
@article{peft-sam2-2025,
  title={Parameter Efficient Fine-Tuning of SAM2 for Interactive Segmentation},
  author={Your Name},
  year={2025}
}
```

---

## 支持与反馈

遇到问题或有改进建议？请：
- 查看 [故障排除](#故障排除) 部分
- 检查 `seg-rl/peft/README.md` 了解更多技术细节
- 提交Issue或Pull Request

---

**祝实验顺利！**

