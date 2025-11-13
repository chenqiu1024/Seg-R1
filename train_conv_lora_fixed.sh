#!/bin/bash

# Conv-LoRA训练 - 修复标签预测问题
# 
# 关键改进：
# 1. label_loss_weight = 1.0 (从0.1增加)
# 2. --use_label_class_weights (处理类别不平衡)
# 3. 正确的学习率调度器
#
# 预期：标签预测准确率从56%提升到95%+

python seg-rl/heatmap/train.py \
  --jsonl outputs/braintumour/heuristic_251108.jsonl \
  --sam_dir outputs/braintumour/sam_masks_heuristic \
  --height 512 --width 512 \
  --arch unet_s \
  --loss kl --sigma 8.0 --tau 1.0 \
  --label_loss_weight 1.0 \
  --use_label_class_weights \
  --batch_size 16 --epochs 80 --amp \
  --lr 1e-4 --weight_decay 1e-4 \
  --grad_clip 1.0 \
  --use_sam_encoder \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --sam_peft_method conv_lora \
  --sam_conv_lora_rank 16 \
  --sam_conv_lora_alpha 32.0 \
  --sam_conv_lora_kernel_size 3 \
  --lr_scheduler warmup_cosine \
  --warmup_epochs 5 \
  --val_ratio 0.1 --test_ratio 0.0 --seed 42 \
  --eval_thresholds "5,10,15,20" \
  --save_every 10 --save_steps 500 --progress \
  --out_dir outputs/braintumour/conv_lora_fixed_label \
  2>&1 | tee outputs/braintumour/conv_lora_fixed_label/train.log

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo ""
echo "训练配置已保存到:"
echo "  outputs/braintumour/conv_lora_fixed_label/training_args.json"
echo ""
echo "推理时使用:"
echo "  python seg-rl/heatmap/predict_point_sequence_with_sam.py \\"
echo "    --model_path outputs/braintumour/conv_lora_fixed_label/model_epoch_80.pt \\"
echo "    --height 512 --width 512 \\"
echo "    --num_points 17 \\"
echo "    ... 其他参数"
echo ""

