# 参数自动记录功能 - 快速使用指南

## 功能说明

从现在开始，训练和推理脚本会自动保存所有参数到 JSON 文件，方便实验管理和复现。

## 文件位置

### 训练脚本 (train.py)
```
<out_dir>/training_args.json
```

### 推理脚本 (predict_point_sequence_with_sam.py)
```
<sam_masks_dir>/inference_args.json           # 推理参数
<sam_masks_dir>/model_training_args.json     # 模型训练参数（自动拷贝）
```

## 快速示例

### 1. 训练模型
```bash
python -m seg-rl.heatmap.train \
  --jsonl data.jsonl \
  --sam_dir masks/ \
  --use_sam_encoder \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --sam_lora_enabled \
  --sam_lora_rank 8 \
  --epochs 100 \
  --out_dir outputs/my_experiment

# ✅ 自动生成: outputs/my_experiment/training_args.json
```

### 2. 查看训练参数
```bash
# 方法1: 直接查看
cat outputs/my_experiment/training_args.json

# 方法2: 格式化查看（需要安装 jq）
jq . outputs/my_experiment/training_args.json

# 方法3: 查看关键参数
jq '{sam_lora_rank, sam_lora_alpha, lr, epochs}' outputs/my_experiment/training_args.json
```

### 3. 推理
```bash
python seg-rl/heatmap/predict_point_sequence_with_sam.py \
  --model_path outputs/my_experiment/model_epoch_100.pt \
  --images_dir test_images/ \
  --masks_dir test_masks/ \
  --output_jsonl outputs/my_pred/results.jsonl \
  --sam_masks_dir outputs/my_pred \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --num_points 17

# ✅ 自动生成: 
#   - outputs/my_pred/inference_args.json
#   - outputs/my_pred/model_training_args.json (从训练目录拷贝)
```

### 4. 查看完整实验链路
```bash
# 查看推理参数
cat outputs/my_pred/inference_args.json

# 查看使用的模型的训练参数
cat outputs/my_pred/model_training_args.json

# 现在您可以完整追溯：
# 训练配置 → 训练结果 → 推理配置 → 推理结果
```

## 参数文件内容示例

### training_args.json
```json
{
  "jsonl": "datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour-251001.jsonl",
  "sam_dir": "datasets/seg_r1_md/Task01_BrainTumour/pretrain_gt_masks-251001",
  "height": 512,
  "width": 512,
  "arch": "unet_s",
  "use_sam_encoder": true,
  "sam_lora_enabled": true,
  "sam_lora_rank": 8,
  "sam_lora_alpha": 16.0,
  "lr": 0.0001,
  "epochs": 100,
  "command": "python -m seg-rl.heatmap.train --jsonl ... --use_sam_encoder ...",
  "timestamp": "2025-11-09T10:30:45.123456"
}
```

### inference_args.json
```json
{
  "model_path": "outputs/my_experiment/model_epoch_100.pt",
  "num_points": 17,
  "resize": [512, 512],
  "device": "cuda",
  "command": "python seg-rl/heatmap/predict_point_sequence_with_sam.py ...",
  "timestamp": "2025-11-09T14:20:30.789012"
}
```

## 快速命令

### 复现训练
```bash
# 从参数文件提取命令并执行
eval $(jq -r '.command' outputs/my_experiment/training_args.json)
```

### 查看关键参数对比
```bash
# 对比两个实验的 LoRA 配置
echo "实验1:"
jq '{sam_lora_rank, sam_lora_alpha}' outputs/exp1/training_args.json
echo "实验2:"
jq '{sam_lora_rank, sam_lora_alpha}' outputs/exp2/training_args.json
```

### 批量查看实验
```bash
# 查看所有实验的 LoRA rank
for f in outputs/*/training_args.json; do
    dir=$(dirname $f)
    rank=$(jq -r '.sam_lora_rank // "N/A"' $f)
    echo "$dir: rank=$rank"
done
```

## 旧实验的处理

### 自动检测

推理脚本会自动检测模型目录是否有 `training_args.json`：
- ✅ 如果有，自动拷贝
- ℹ️ 如果没有，会打印提示信息但继续运行

### 手动补充

对于旧的训练目录，您可以手动创建参数文件：

```bash
# 创建基本的参数记录
cat > outputs/old_experiment/training_args.json << 'EOF'
{
  "note": "Manually created for old experiment",
  "epochs": 100,
  "lr": 0.0001,
  "use_sam_encoder": false,
  "timestamp": "2025-11-08T00:00:00"
}
EOF
```

## 注意事项

1. **参数文件会被覆盖**: 每次运行都会覆盖同名文件
2. **需要 jq 工具**: 建议安装 `jq` 以方便查看 JSON
   ```bash
   apt-get install jq  # Ubuntu/Debian
   brew install jq     # macOS
   ```
3. **路径信息**: JSON 包含完整路径，分享时注意脱敏

## 总结

✅ **自动记录**: 无需手动记录参数  
✅ **完整追溯**: 从训练到推理的完整链路  
✅ **易于复现**: 一键获取运行命令  
✅ **批量分析**: JSON 格式便于程序处理  
✅ **向后兼容**: 旧实验仍可正常运行  

详细文档见：`EXPERIMENT_TRACKING.md`

