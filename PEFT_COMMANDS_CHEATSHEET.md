# PEFT命令速查表

快速参考：常用命令的简化版本

## 数据生成

```bash
# 生成第一个点
python seg-rl/annotator/gen_point_jsonl_from_masks.py \
  --images_dir datasets/YOUR_DATASET/images \
  --masks_dir datasets/YOUR_DATASET/masks \
  --output_jsonl datasets/YOUR_DATASET/train.jsonl

# 生成SAM掩模
python seg-rl/sam2_segment_from_points.py \
  --input_jsonl datasets/YOUR_DATASET/train.jsonl \
  --json_output datasets/YOUR_DATASET/train.jsonl \
  --output_dir datasets/YOUR_DATASET/sam_masks \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --device cuda --resize 512 512 --skip_existing

# 追加下一个点
python seg-rl/annotator/gen_point_jsonl_from_masks.py \
  --appendto_jsonl datasets/YOUR_DATASET/train.jsonl

# 重复上述两步，迭代生成多个点
```

## 监督训练

### 最小配置（快速测试）
```bash
python -m seg-rl.peft.train_supervised_peft \
  --jsonl datasets/YOUR_DATASET/train.jsonl \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --out_dir outputs/peft_test \
  --epochs 10 --batch_size 4 --device cuda
```

### 推荐配置（完整训练）
```bash
python -m seg-rl.peft.train_supervised_peft \
  --jsonl datasets/YOUR_DATASET/train.jsonl \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --lora_rank 16 --lora_alpha 32 \
  --feature_scale 8 --fusion_mode film \
  --image_size 512 512 \
  --loss kl --sigma 8.0 \
  --batch_size 8 --epochs 40 --amp \
  --lr_sam 1e-5 --lr_point 1e-4 \
  --out_dir outputs/peft_supervised \
  --save_every 5 --auto_resume --tb \
  --device cuda
```

### 仅训练点网络（SAM冻结）
```bash
python -m seg-rl.peft.train_supervised_peft \
  --jsonl datasets/YOUR_DATASET/train.jsonl \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --freeze_sam \
  --lr_point 1e-4 \
  --out_dir outputs/peft_frozen \
  --epochs 20 --device cuda
```

## GRPO训练

### 最小配置
```bash
python -m seg-rl.peft.train_grpo_peft \
  --train_json datasets/YOUR_DATASET/train.jsonl \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --init_policy outputs/peft_supervised/checkpoint_best.pt \
  --out_dir outputs/peft_grpo \
  --epochs 3 --batch_size 4 --device cuda
```

### 推荐配置
```bash
python -m seg-rl.peft.train_grpo_peft \
  --train_json datasets/YOUR_DATASET/train.jsonl \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --init_policy outputs/peft_supervised/checkpoint_best.pt \
  --out_dir outputs/peft_grpo \
  --height 512 --width 512 --stride 8 \
  --max_points 16 --group_size 4 \
  --epochs 5 --batch_size 8 \
  --lr_sam 5e-6 --lr_point 5e-5 \
  --beta_kl 0.02 --beta_entropy 0.01 \
  --save_every 100 --tb --device cuda
```

## 评估

```bash
# 评估任意checkpoint
python -m seg-rl.peft.eval_peft_model \
  --test_json datasets/YOUR_DATASET/test.jsonl \
  --checkpoint outputs/peft_supervised/checkpoint_best.pt \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --out_dir outputs/eval \
  --device cuda
```

## 可视化

```bash
# SAM特征可视化
python -m seg-rl.visualization.viz_sam_features \
  --image path/to/image.jpg \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --lora_checkpoint outputs/peft_supervised/checkpoint_best.pt \
  --out_dir outputs/viz/features

# 预测热力图可视化
python -m seg-rl.visualization.viz_peft_predictions \
  --image path/to/image.jpg \
  --prev_mask path/to/prev_mask.png \
  --checkpoint outputs/peft_supervised/checkpoint_best.pt \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --out_dir outputs/viz/predictions
```

## TensorBoard

```bash
# 启动TensorBoard
tensorboard --logdir outputs/peft_supervised/tensorboard --port 6006

# 在浏览器打开
# http://localhost:6006
```

## 常用参数快速调整

### GPU显存不足？
```bash
--batch_size 4        # 减小batch size
--image_size 256 256  # 减小图像尺寸
--amp                 # 启用混合精度
```

### 加快训练？
```bash
--epochs 20           # 减少epoch数
--num_workers 8       # 增加数据加载线程
--batch_size 16       # 增大batch size（如果显存够）
```

### 调试模式？
```bash
--freeze_sam          # 冻结SAM
--epochs 5            # 少量epoch
--batch_size 2        # 小batch
--val_ratio 0.2       # 更多验证数据
```

## 目录结构

训练后的输出目录结构：

```
outputs/
└── peft_supervised/
    ├── checkpoint_best.pt           # 最佳模型
    ├── checkpoint_epoch005.pt
    ├── checkpoint_epoch010.pt
    ├── ...
    └── tensorboard/                 # TensorBoard日志
        └── events.out.tfevents.*
```

## 环境变量

```bash
# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0

# 设置显存增长（TensorFlow后端，如果使用）
export TF_FORCE_GPU_ALLOW_GROWTH=true

# PyTorch显存优化
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

## 快速诊断

```bash
# 检查数据
python -c "import json; d=json.load(open('datasets/YOUR_DATASET/train.jsonl')); print(f'Samples: {len(d)}, Points/sample: {len(d[0][\"points\"])}')"

# 检查checkpoint
python -c "import torch; c=torch.load('outputs/peft_supervised/checkpoint_best.pt', map_location='cpu'); print(f'Epoch: {c[\"epoch\"]}, Metrics: {c[\"metrics\"]}')"

# 检查GPU
nvidia-smi

# 清理GPU缓存
python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')"
```

---

详细文档请参考：`README_PEFT_EXPERIMENT_GUIDE.md`

