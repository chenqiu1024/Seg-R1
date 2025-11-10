# 实验参数自动保存功能 - 实现总结

## ✅ 已完成的修改

### 1. 修改 `train.py`

**位置**: 在 `main()` 函数开始，创建输出目录后立即保存参数

**添加的代码**:
```python
# 保存训练参数到 JSON 文件
import sys
args_dict = vars(args).copy()
args_dict['command'] = ' '.join(sys.argv)  # 保存完整命令
args_dict['timestamp'] = __import__('datetime').datetime.now().isoformat()
args_json_path = os.path.join(args.out_dir, "training_args.json")
with open(args_json_path, 'w', encoding='utf-8') as f:
    json.dump(args_dict, f, indent=2, ensure_ascii=False)
print(f"[Config] Training arguments saved to {args_json_path}")
```

**效果**:
- ✅ 每次训练自动生成 `<out_dir>/training_args.json`
- ✅ 包含所有命令行参数、完整命令和时间戳
- ✅ 可用于实验复现和参数查询

### 2. 修改 `predict_point_sequence_with_sam.py`

**位置**: 在 `main()` 函数中，创建输出目录后

**添加的代码**:
```python
# 保存推理参数到 JSON 文件
import sys
import json
import shutil
from datetime import datetime

inference_args_dict = vars(args).copy()
inference_args_dict['command'] = ' '.join(sys.argv)
inference_args_dict['timestamp'] = datetime.now().isoformat()
inference_args_path = os.path.join(args.sam_masks_dir, "inference_args.json")
with open(inference_args_path, 'w', encoding='utf-8') as f:
    json.dump(inference_args_dict, f, indent=2, ensure_ascii=False)
print(f"[Config] Inference arguments saved to {inference_args_path}")

# 如果模型路径所在目录下存在训练参数文件，拷贝到输出目录
if args.model_path and os.path.isfile(args.model_path):
    model_dir = os.path.dirname(args.model_path)
    training_args_src = os.path.join(model_dir, "training_args.json")
    if os.path.isfile(training_args_src):
        training_args_dst = os.path.join(args.sam_masks_dir, "model_training_args.json")
        try:
            shutil.copy2(training_args_src, training_args_dst)
            print(f"[Config] Copied training arguments from {training_args_src}")
            print(f"[Config]   to {training_args_dst}")
        except Exception as e:
            print(f"[Warning] Failed to copy training arguments: {e}")
    else:
        print(f"[Info] No training_args.json found in model directory: {model_dir}")
```

**效果**:
- ✅ 每次推理自动生成 `<sam_masks_dir>/inference_args.json`
- ✅ 自动拷贝模型的训练参数到 `<sam_masks_dir>/model_training_args.json`
- ✅ 实现完整的训练-推理链路追溯

### 3. 新增工具脚本 `show_experiment_config.py`

**功能**: 友好地查看和对比实验配置

**用法**:
```bash
# 查看单个配置
python seg-rl/heatmap/show_experiment_config.py outputs/exp1/training_args.json

# 对比两个配置
python seg-rl/heatmap/show_experiment_config.py \
    outputs/exp1/training_args.json \
    outputs/exp2/training_args.json
```

### 4. 新增文档

- ✅ `EXPERIMENT_TRACKING.md` - 完整的功能说明和使用指南
- ✅ `PARAM_TRACKING_USAGE.md` - 快速使用指南
- ✅ `PARAM_SAVING_SUMMARY.md` - 本文档

## 文件结构

### 训练后的目录结构

```
outputs/braintumour/heatmap_train-latelora-251109/
├── training_args.json          # ← 新增：训练参数
├── model_epoch_5.pt
├── model_epoch_10.pt
├── model_epoch_15.pt
├── last.pt
├── plots/
│   ├── loss.png
│   └── pck.png
└── vis/
    └── val_or_train_vis.png
```

### 推理后的目录结构

```
outputs/braintumour/pred_supervised-r8-epoch15-251109/
├── inference_args.json         # ← 新增：推理参数
├── model_training_args.json    # ← 新增：模型训练参数（自动拷贝）
├── BRATS_001_z0029/
│   ├── 0.png
│   ├── 1.png
│   └── ...
└── ...
```

## 使用示例

### 示例 1: 训练并记录参数

```bash
python -m seg-rl.heatmap.train \
  --jsonl datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour-251001.jsonl \
  --sam_dir datasets/seg_r1_md/Task01_BrainTumour/pretrain_gt_masks-251001 \
  --height 512 --width 512 \
  --arch unet_s \
  --use_sam_encoder \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --sam_lora_enabled \
  --sam_lora_rank 8 \
  --sam_lora_alpha 16.0 \
  --epochs 100 \
  --out_dir outputs/braintumour/exp_r8_alpha16

# 输出:
# [Config] Training arguments saved to outputs/braintumour/exp_r8_alpha16/training_args.json

# 查看保存的参数:
cat outputs/braintumour/exp_r8_alpha16/training_args.json | jq '.'
```

### 示例 2: 推理并记录参数

```bash
python seg-rl/heatmap/predict_point_sequence_with_sam.py \
  --model_path outputs/braintumour/exp_r8_alpha16/model_epoch_100.pt \
  --images_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/images \
  --masks_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/masks \
  --output_jsonl outputs/braintumour/pred_r8_n17/results.jsonl \
  --sam_masks_dir outputs/braintumour/pred_r8_n17 \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --num_points 17 \
  --device cuda \
  --resize 512 512

# 输出:
# [Config] Inference arguments saved to outputs/braintumour/pred_r8_n17/inference_args.json
# [Config] Copied training arguments from outputs/braintumour/exp_r8_alpha16/training_args.json
# [Config]   to outputs/braintumour/pred_r8_n17/model_training_args.json

# 现在可以完整追溯实验链路
```

### 示例 3: 查看配置

```bash
# 使用工具脚本查看（格式化输出）
python seg-rl/heatmap/show_experiment_config.py \
  outputs/braintumour/exp_r8_alpha16/training_args.json

# 或直接用 jq 查看
jq . outputs/braintumour/exp_r8_alpha16/training_args.json

# 查看关键参数
jq '{sam_lora_rank, sam_lora_alpha, lr, epochs}' \
  outputs/braintumour/exp_r8_alpha16/training_args.json
```

### 示例 4: 对比实验

```bash
# 对比两个实验的配置差异
python seg-rl/heatmap/show_experiment_config.py \
  outputs/braintumour/exp_r8_alpha16/training_args.json \
  outputs/braintumour/exp_r16_alpha32/training_args.json

# 或使用 diff
diff \
  <(jq -S . outputs/braintumour/exp_r8_alpha16/training_args.json) \
  <(jq -S . outputs/braintumour/exp_r16_alpha32/training_args.json)
```

### 示例 5: 复现实验

```bash
# 提取命令并执行
eval $(jq -r '.command' outputs/braintumour/exp_r8_alpha16/training_args.json)
```

## 参数文件格式

### training_args.json 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `jsonl` | string | 训练数据路径 |
| `sam_dir` | string | SAM masks 目录 |
| `height` | int | 图像高度 |
| `width` | int | 图像宽度 |
| `arch` | string | 架构类型 |
| `use_sam_encoder` | bool | 是否使用 SAM encoder |
| `sam_checkpoint` | string | SAM checkpoint 路径 |
| `sam_lora_enabled` | bool | 是否启用 LoRA |
| `sam_lora_rank` | int | LoRA 秩 |
| `sam_lora_alpha` | float | LoRA alpha |
| `epochs` | int | 训练轮数 |
| `lr` | float | 学习率 |
| `command` | string | 完整执行命令 |
| `timestamp` | string | ISO 8601 格式时间戳 |

### inference_args.json 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `model_path` | string | 模型 checkpoint 路径 |
| `images_dir` | string | 图像目录 |
| `masks_dir` | string | 掩模目录 |
| `output_jsonl` | string | 输出 JSONL 路径 |
| `sam_masks_dir` | string | SAM masks 输出目录 |
| `num_points` | int | 点序列长度 |
| `device` | string | 设备类型 |
| `resize` | list | resize 尺寸 [W, H] |
| `command` | string | 完整执行命令 |
| `timestamp` | string | ISO 8601 格式时间戳 |

## 向后兼容性

### 对旧实验的影响

- ✅ **旧训练目录**: 不受影响，仍可正常使用
- ✅ **旧推理流程**: 仍可正常运行
- ℹ️ **缺少参数文件**: 推理时会提示 "No training_args.json found"，但继续运行

### 为旧实验补充参数文件

如果您想为旧实验添加参数文件（可选）：

```bash
# 手动创建 training_args.json
cat > outputs/old_experiment/training_args.json << 'EOF'
{
  "note": "手动创建的参数记录",
  "use_sam_encoder": false,
  "epochs": 80,
  "lr": 0.0001,
  "arch": "unet_s",
  "timestamp": "2025-11-08T00:00:00"
}
EOF
```

## 实用技巧

### 技巧 1: 快速查看所有实验

```bash
# 列出所有训练实验及其关键参数
for dir in outputs/braintumour/*/; do
    if [ -f "$dir/training_args.json" ]; then
        echo "📁 $dir"
        jq -r '"\tLoRA: \(.sam_lora_enabled) (rank=\(.sam_lora_rank))\tEpochs: \(.epochs)\tLR: \(.lr)"' \
          "$dir/training_args.json"
    fi
done
```

### 技巧 2: 创建实验表格

```python
#!/usr/bin/env python3
"""生成实验对比表格"""

import json
import glob
from pathlib import Path

experiments = []
for path in sorted(glob.glob("outputs/**/training_args.json", recursive=True)):
    with open(path) as f:
        config = json.load(f)
    
    exp_dir = str(Path(path).parent.name)
    experiments.append({
        'name': exp_dir,
        'sam_lora': config.get('sam_lora_enabled', False),
        'rank': config.get('sam_lora_rank', 'N/A'),
        'alpha': config.get('sam_lora_alpha', 'N/A'),
        'lr': config.get('lr', 'N/A'),
        'epochs': config.get('epochs', 'N/A'),
        'timestamp': config.get('timestamp', 'N/A')[:10],  # 只取日期
    })

# 打印表格
print(f"{'实验名称':<40} {'LoRA':<8} {'Rank':<6} {'Alpha':<8} {'LR':<10} {'Epochs':<8} {'日期':<12}")
print("="*110)
for exp in experiments:
    print(f"{exp['name']:<40} {str(exp['sam_lora']):<8} {str(exp['rank']):<6} "
          f"{str(exp['alpha']):<8} {str(exp['lr']):<10} {str(exp['epochs']):<8} {exp['timestamp']:<12}")
```

### 技巧 3: 实验版本控制

```bash
# .gitignore 中添加
outputs/**/*.pt              # 不提交大模型文件
outputs/**/*.png             # 不提交图片
outputs/**/*.jsonl           # 不提交数据
!outputs/**/training_args.json        # 但保留参数文件
!outputs/**/inference_args.json
!outputs/**/model_training_args.json
```

## 调试和验证

### 验证功能是否正常

```bash
# 运行一个小型训练测试
python -m seg-rl.heatmap.train \
  --jsonl test_data.jsonl \
  --sam_dir test_masks/ \
  --height 240 --width 240 \
  --epochs 1 \
  --batch_size 2 \
  --out_dir outputs/test_param_saving

# 检查是否生成了参数文件
ls -lh outputs/test_param_saving/training_args.json

# 查看内容
cat outputs/test_param_saving/training_args.json | jq '.'

# 清理测试
rm -rf outputs/test_param_saving/
```

## 与其他功能的集成

### 与 checkpoint 元数据的关系

现在我们有两层参数记录：

1. **Checkpoint 内部元数据** (在 .pt 文件中):
   - 用于模型加载时的自动类型检测
   - 包含核心模型配置（use_sam_encoder, sam_lora_enabled等）
   
2. **外部 JSON 参数文件** (training_args.json):
   - 用于人类可读的实验记录
   - 包含完整的训练配置和命令
   - 便于查询、对比和复现

两者互补，各有用途：
- Checkpoint 元数据 → 程序自动化使用
- JSON 参数文件 → 人工管理和分析

### 与日志系统的关系

参数文件补充了日志系统：

```
实验完整记录 = 参数文件 + 训练日志 + checkpoint + 结果
```

建议同时保存训练日志：

```bash
python -m seg-rl.heatmap.train \
  --out_dir outputs/exp001 \
  ... 其他参数 ... \
  2>&1 | tee outputs/exp001/train.log
```

## 已知限制

1. **参数文件覆盖**: 
   - 每次运行覆盖同名文件
   - 使用 `--auto_resume` 时会更新为最新参数

2. **路径绝对化**:
   - 某些路径可能被转换为绝对路径
   - 跨机器复现时需要调整路径

3. **大型参数**:
   - 某些参数可能很长（如完整命令）
   - 建议使用 `jq` 查看而不是直接 `cat`

## 未来改进方向

### 可能的增强

1. **版本化参数保存**:
   ```
   training_args.json
   training_args.json.backup_20251109
   ```

2. **参数验证**:
   - 检查参数合理性
   - 给出优化建议

3. **自动实验报告**:
   - 从参数文件和结果自动生成报告
   - Markdown 或 HTML 格式

4. **实验数据库**:
   - 将参数存入 SQLite
   - 支持复杂查询和统计

5. **参数模板**:
   - 预定义常用配置
   - 快速启动实验

## 总结

通过这次改进：

✅ **训练脚本**: 自动保存 `training_args.json`  
✅ **推理脚本**: 自动保存 `inference_args.json` 和拷贝 `model_training_args.json`  
✅ **工具脚本**: 提供友好的查看和对比工具  
✅ **文档**: 完整的使用指南和示例  
✅ **向后兼容**: 不影响任何现有功能  

现在您可以：
- 🔍 轻松查询任何实验的配置
- 🔄 一键复现任何实验
- 📊 批量分析多个实验
- 🔗 完整追溯训练-推理链路
- 🤝 方便地与他人分享实验配置

享受更高效的实验管理！🎉

---

**实现日期**: 2025-11-09  
**修改文件**: `train.py`, `predict_point_sequence_with_sam.py`  
**新增工具**: `show_experiment_config.py`  
**状态**: ✅ 已完成并可用

