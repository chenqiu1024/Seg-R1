# 实验参数自动记录功能 - 功能说明

## 📝 功能概述

为了便于记录每次实验使用的各项设置，我们为训练和推理脚本添加了自动参数保存功能。

## ✅ 实现的功能

### 1. 训练参数自动保存 (`train.py`)

**保存位置**: `<out_dir>/training_args.json`

**包含内容**:
- 所有命令行参数
- 完整的执行命令
- 时间戳

**示例**:
```bash
python -m seg-rl.heatmap.train \
  --jsonl data.jsonl \
  --sam_dir masks/ \
  --use_sam_encoder \
  --sam_lora_enabled \
  --sam_lora_rank 8 \
  --out_dir outputs/exp001

# 自动生成: outputs/exp001/training_args.json
```

### 2. 推理参数自动保存 (`predict_point_sequence_with_sam.py`)

**保存位置**: 
- `<sam_masks_dir>/inference_args.json` - 推理参数
- `<sam_masks_dir>/model_training_args.json` - 模型训练参数（自动拷贝）

**功能**:
- 保存推理参数
- 自动从模型目录拷贝训练参数
- 实现完整的训练-推理链路追溯

**示例**:
```bash
python seg-rl/heatmap/predict_point_sequence_with_sam.py \
  --model_path outputs/exp001/model_epoch_100.pt \
  --output_jsonl outputs/pred001/results.jsonl \
  --sam_masks_dir outputs/pred001 \
  --num_points 17 \
  ...

# 自动生成:
#   outputs/pred001/inference_args.json          ← 推理参数
#   outputs/pred001/model_training_args.json     ← 训练参数（拷贝）
```

## 📁 文件结构

### 训练输出目录
```
outputs/exp001/
├── training_args.json     ← 新增：训练参数
├── model_epoch_*.pt
├── last.pt
└── plots/
```

### 推理输出目录
```
outputs/pred001/
├── inference_args.json        ← 新增：推理参数
├── model_training_args.json   ← 新增：模型训练参数（拷贝）
└── BRATS_*/
```

## 🚀 快速使用

### 查看参数
```bash
# 查看训练参数
cat outputs/exp001/training_args.json | jq '.'

# 查看推理参数
cat outputs/pred001/inference_args.json | jq '.'

# 使用工具脚本（更友好的格式）
python seg-rl/heatmap/show_experiment_config.py outputs/exp001/training_args.json
```

### 复现实验
```bash
# 提取并执行命令
eval $(jq -r '.command' outputs/exp001/training_args.json)
```

### 对比实验
```bash
# 对比两个实验的配置
python seg-rl/heatmap/show_experiment_config.py \
  outputs/exp1/training_args.json \
  outputs/exp2/training_args.json
```

## 🔗 完整实验链路追溯

现在您可以完整追溯每个推理结果：

```
1. 查看推理输出目录
   outputs/pred001/

2. 查看推理参数
   → inference_args.json
   
3. 找到使用的模型
   → model_training_args.json (model_path)
   
4. 查看模型训练配置
   → model_training_args.json
   
5. 追溯到训练数据
   → jsonl, sam_dir
```

完整链路：**数据 → 训练配置 → 模型 → 推理配置 → 结果**

## 📚 相关文档

- 📖 **完整指南**: `EXPERIMENT_TRACKING.md`
- 🚀 **快速开始**: `PARAM_TRACKING_USAGE.md`
- 📋 **实现细节**: `PARAM_SAVING_SUMMARY.md`
- 🔧 **工具脚本**: `show_experiment_config.py`

## 💡 优势

1. **自动化**: 无需手动记录参数
2. **完整性**: 包含所有参数和命令
3. **可复现**: 一键重现任何实验
4. **可追溯**: 完整的实验链路
5. **易分析**: JSON 格式便于处理
6. **向后兼容**: 不影响现有功能

## 🎯 现在就开始使用

下次训练或推理时，参数会自动保存。您可以：

```bash
# 训练后查看
cat <out_dir>/training_args.json

# 推理后查看
cat <sam_masks_dir>/inference_args.json
cat <sam_masks_dir>/model_training_args.json
```

就这么简单！🎉

