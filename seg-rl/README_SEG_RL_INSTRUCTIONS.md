# Seg-RL指南
## 完整训练-推理流程
### 预训练
#### 1. 启发式生成提示点序列和分割图像，作为监督学习阶段训练提示点预测模型的训练数据
```
python seg-rl/annotator/gen_point_sequence_with_sam.py \
  --images_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/images \
  --masks_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/masks \
  --output_jsonl outputs/braintumour/heuristic_251108.jsonl \
  --sam_masks_dir outputs/braintumour/sam_masks_heuristic \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --device cuda \
  --resize 512 512 \
  --num_points 17
```

#### 2. 评估启发式方法生成预测分割masks的质量
```
python seg-rl/evaluation/eval_sam_masks.py    --input_json outputs/braintumour/heuristic_251108.jsonl --max_prompts 17    --output_plot outputs/braintumour/heuristic_metrics_curve-251108.png
```

#### 3. 监督学习阶段，用启发式方法得到的数据去训练提示点预测模型
```
python -m seg-rl.heatmap.train \
  --jsonl outputs/braintumour/heuristic_251108.jsonl \
  --sam_dir outputs/braintumour/sam_masks_heuristic \
  --height 512 --width 512 --arch unet_s \
  --loss kl --sigma 10.0 --tau 1.2 \
  --eval_thresholds "10,15,20,25" \
  --label_loss_weight 0.2 \
  --batch_size 8 --epochs 80 --amp \
  --lr 1e-4 --weight_decay 1e-4 --grad_clip 1.0 \
  --lr_scheduler cosine \
  --val_ratio 0.1 --test_ratio 0.1 --seed 42 \
  --save_every 5 --save_steps 500 --progress --auto_resume \
  --out_dir outputs/braintumour/heatmap_train-251108-hires
``` 

#### 4. 用监督训练得到的提示点预测模型作迭代式SAM分割
```
python seg-rl/heatmap/predict_point_sequence_with_sam.py \
  --model_path outputs/braintumour/.pt \
  --images_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/images \
  --masks_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/masks \
  --output_jsonl outputs/braintumour/pred_supervised-251108.jsonl \
  --sam_masks_dir outputs/braintumour/pred_supervised-251108 \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --device cuda \
  --resize 512 512 \
  --num_points 17
```

#### 5. 评估提示点预测模型生成预测分割masks的质量
```
python seg-rl/evaluation/eval_sam_masks.py \ 
   --input_json outputs/braintumour/pred_supervised-251108.jsonl \ --max_prompts 17 \
   --output_plot outputs/braintumour/supervised_metrics_curve-251108.png
```

#### 6. 可视化提示点预测效果
python -m seg-rl.visualization.viz_training_data \
        --jsonl outputs/braintumour/pred_supervised-80last-251108.jsonl \
        --out_dir outputs/braintumour/viz-pred_supervised-80last-251108 \
        --mode all \
        --show_prediction