# 参数自动记录 - 快速开始

## 🎯 一句话说明

从现在开始，训练和推理时会自动保存所有参数到 JSON 文件，无需任何额外操作。

## ⚡ 快速示例

### 训练
```bash
python -m seg-rl.heatmap.train \
  --jsonl data.jsonl \
  --sam_dir masks/ \
  --out_dir outputs/my_experiment \
  ... 其他参数 ...

# ✅ 自动生成: outputs/my_experiment/training_args.json
```

### 查看训练参数
```bash
# 简单查看
cat outputs/my_experiment/training_args.json

# 格式化查看（需要 jq）
jq . outputs/my_experiment/training_args.json

# 使用工具脚本（推荐）
python seg-rl/heatmap/show_experiment_config.py \
  outputs/my_experiment/training_args.json
```

### 推理
```bash
python seg-rl/heatmap/predict_point_sequence_with_sam.py \
  --model_path outputs/my_experiment/model_epoch_100.pt \
  --sam_masks_dir outputs/my_pred \
  --num_points 17 \
  ... 其他参数 ...

# ✅ 自动生成: 
#   outputs/my_pred/inference_args.json          (推理参数)
#   outputs/my_pred/model_training_args.json     (训练参数，自动拷贝)
```

### 完整追溯
```bash
# 从推理结果追溯到训练配置
cat outputs/my_pred/model_training_args.json | jq .
```

## 📂 生成的文件

### 训练输出
```
<out_dir>/
└── training_args.json    ← 新增
```

### 推理输出
```
<sam_masks_dir>/
├── inference_args.json        ← 新增
└── model_training_args.json   ← 新增（自动拷贝）
```

## 🎁 无需任何改变

- ✅ 不需要添加新参数
- ✅ 不需要修改现有命令
- ✅ 完全自动化
- ✅ 向后兼容

## 📖 详细文档

- `PARAM_TRACKING_USAGE.md` - 使用指南
- `EXPERIMENT_TRACKING.md` - 完整文档

## 🧪 测试

```bash
python seg-rl/heatmap/test_param_saving.py
# ✅ 所有测试通过 (3/3)
```

---

就这么简单！享受自动化的实验管理！🎉

