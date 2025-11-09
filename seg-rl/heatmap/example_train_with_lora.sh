#!/bin/bash

# SAM Late LoRA 训练示例脚本
# 
# 此脚本演示如何使用 SAM image encoder 配合 Late LoRA 进行训练
# 请根据您的实际路径和需求修改相关参数

# ============================================================
# 配置参数
# ============================================================

# 数据路径
JSONL_PATH="datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour-251001.jsonl"
SAM_DIR="datasets/seg_r1_md/Task01_BrainTumour/pretrain_gt_masks-251001"

# SAM checkpoint 路径
SAM_CHECKPOINT="third_party/sam2/checkpoints/sam2.1_hiera_large.pt"

# 输出目录
OUTPUT_DIR="outputs/braintumour/sam_lora_train_example"

# 训练参数
EPOCHS=100
BATCH_SIZE=16
HEIGHT=240
WIDTH=240

# LoRA 参数
LORA_RANK=8
LORA_ALPHA=16.0
LORA_DROPOUT=0.0

# ============================================================
# 示例 1: 标准训练（不使用 SAM）- 作为基线
# ============================================================

echo "=========================================="
echo "示例 1: 标准训练（不使用 SAM）"
echo "=========================================="

python -m seg-rl.heatmap.train \
  --jsonl ${JSONL_PATH} \
  --sam_dir ${SAM_DIR} \
  --height ${HEIGHT} --width ${WIDTH} \
  --arch unet_s \
  --loss kl --sigma 8.0 --tau 1.0 \
  --label_loss_weight 0.1 \
  --batch_size ${BATCH_SIZE} \
  --epochs ${EPOCHS} \
  --amp \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --grad_clip 1.0 \
  --lr_scheduler warmup_cosine \
  --warmup_epochs 3 \
  --val_ratio 0.1 \
  --test_ratio 0.1 \
  --seed 42 \
  --save_every 10 \
  --save_steps 500 \
  --progress \
  --auto_resume \
  --vis_mode sample \
  --vis_count 16 \
  --out_dir ${OUTPUT_DIR}_standard

echo "✓ 标准训练完成"
echo ""

# ============================================================
# 示例 2: 使用 SAM encoder（冻结）
# ============================================================

echo "=========================================="
echo "示例 2: 使用 SAM encoder（冻结）"
echo "=========================================="

# 检查 SAM checkpoint 是否存在
if [ ! -f "${SAM_CHECKPOINT}" ]; then
    echo "错误：SAM checkpoint 不存在: ${SAM_CHECKPOINT}"
    echo "请先下载 SAM checkpoint："
    echo "  mkdir -p third_party/sam2/checkpoints"
    echo "  wget -O ${SAM_CHECKPOINT} https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
    exit 1
fi

python -m seg-rl.heatmap.train \
  --jsonl ${JSONL_PATH} \
  --sam_dir ${SAM_DIR} \
  --height ${HEIGHT} --width ${WIDTH} \
  --arch unet_s \
  --loss kl --sigma 8.0 --tau 1.0 \
  --label_loss_weight 0.1 \
  --batch_size ${BATCH_SIZE} \
  --epochs ${EPOCHS} \
  --amp \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --grad_clip 1.0 \
  --lr_scheduler warmup_cosine \
  --warmup_epochs 3 \
  --val_ratio 0.1 \
  --test_ratio 0.1 \
  --seed 42 \
  --save_every 10 \
  --save_steps 500 \
  --progress \
  --auto_resume \
  --vis_mode sample \
  --vis_count 16 \
  --use_sam_encoder \
  --sam_checkpoint ${SAM_CHECKPOINT} \
  --out_dir ${OUTPUT_DIR}_sam_frozen

echo "✓ SAM 冻结训练完成"
echo ""

# ============================================================
# 示例 3: 使用 SAM encoder + Late LoRA（推荐）
# ============================================================

echo "=========================================="
echo "示例 3: 使用 SAM encoder + Late LoRA"
echo "=========================================="

python -m seg-rl.heatmap.train \
  --jsonl ${JSONL_PATH} \
  --sam_dir ${SAM_DIR} \
  --height ${HEIGHT} --width ${WIDTH} \
  --arch unet_s \
  --loss kl --sigma 8.0 --tau 1.0 \
  --label_loss_weight 0.1 \
  --batch_size ${BATCH_SIZE} \
  --epochs ${EPOCHS} \
  --amp \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --grad_clip 1.0 \
  --lr_scheduler warmup_cosine \
  --warmup_epochs 3 \
  --val_ratio 0.1 \
  --test_ratio 0.1 \
  --seed 42 \
  --save_every 10 \
  --save_steps 500 \
  --progress \
  --auto_resume \
  --vis_mode sample \
  --vis_count 16 \
  --use_sam_encoder \
  --sam_checkpoint ${SAM_CHECKPOINT} \
  --sam_lora_enabled \
  --sam_lora_rank ${LORA_RANK} \
  --sam_lora_alpha ${LORA_ALPHA} \
  --sam_lora_dropout ${LORA_DROPOUT} \
  --out_dir ${OUTPUT_DIR}_sam_lora

echo "✓ SAM + LoRA 训练完成"
echo ""

# ============================================================
# 示例 4: 使用独立的 LoRA 学习率
# ============================================================

echo "=========================================="
echo "示例 4: 使用独立的 LoRA 学习率"
echo "=========================================="

python -m seg-rl.heatmap.train \
  --jsonl ${JSONL_PATH} \
  --sam_dir ${SAM_DIR} \
  --height ${HEIGHT} --width ${WIDTH} \
  --arch unet_s \
  --loss kl --sigma 8.0 --tau 1.0 \
  --label_loss_weight 0.1 \
  --batch_size ${BATCH_SIZE} \
  --epochs ${EPOCHS} \
  --amp \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --grad_clip 1.0 \
  --lr_scheduler warmup_cosine \
  --warmup_epochs 3 \
  --val_ratio 0.1 \
  --test_ratio 0.1 \
  --seed 42 \
  --save_every 10 \
  --save_steps 500 \
  --progress \
  --auto_resume \
  --vis_mode sample \
  --vis_count 16 \
  --use_sam_encoder \
  --sam_checkpoint ${SAM_CHECKPOINT} \
  --sam_lora_enabled \
  --sam_lora_rank ${LORA_RANK} \
  --sam_lora_alpha ${LORA_ALPHA} \
  --sam_lora_dropout ${LORA_DROPOUT} \
  --sam_lora_lr 5e-5 \
  --out_dir ${OUTPUT_DIR}_sam_lora_separate_lr

echo "✓ 独立学习率训练完成"
echo ""

# ============================================================
# 完成
# ============================================================

echo "=========================================="
echo "所有训练示例完成！"
echo "=========================================="
echo ""
echo "输出目录："
echo "  1. 标准模型: ${OUTPUT_DIR}_standard"
echo "  2. SAM 冻结: ${OUTPUT_DIR}_sam_frozen"
echo "  3. SAM + LoRA: ${OUTPUT_DIR}_sam_lora"
echo "  4. 独立学习率: ${OUTPUT_DIR}_sam_lora_separate_lr"
echo ""
echo "您可以比较不同方法的性能："
echo "  - 查看训练曲线: outputs/<dir>/plots/"
echo "  - 查看可视化结果: outputs/<dir>/vis/"
echo "  - 查看 checkpoint: outputs/<dir>/*.pt"
echo ""
echo "进行推理："
echo "  python -m seg-rl.heatmap.infer \\"
echo "    --images <image_dir> \\"
echo "    --ckpt outputs/<dir>/model_epoch_100.pt \\"
echo "    --height ${HEIGHT} --width ${WIDTH} \\"
echo "    --sam_checkpoint ${SAM_CHECKPOINT}  # 仅 SAM-based 模型需要"
echo ""

