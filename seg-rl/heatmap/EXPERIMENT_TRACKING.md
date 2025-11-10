# 实验参数自动记录功能

## 概述

为了便于实验管理和结果复现，训练和推理脚本现在会自动保存所有命令行参数到 JSON 文件。

## 功能说明

### 1. 训练参数记录 (`train.py`)

#### 自动保存位置
```
<out_dir>/training_args.json
```

#### 保存内容
- ✅ 所有命令行参数（argparse 解析的所有参数）
- ✅ 完整的执行命令（可直接复制运行）
- ✅ 时间戳（记录训练开始时间）

#### 示例输出

```json
{
  "jsonl": "datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour-251001.jsonl",
  "sam_dir": "datasets/seg_r1_md/Task01_BrainTumour/pretrain_gt_masks-251001",
  "height": 512,
  "width": 512,
  "arch": "unet_s",
  "loss": "kl",
  "sigma": 8.0,
  "tau": 1.0,
  "batch_size": 16,
  "epochs": 100,
  "lr": 0.0001,
  "use_sam_encoder": true,
  "sam_checkpoint": "third_party/sam2/checkpoints/sam2.1_hiera_large.pt",
  "sam_lora_enabled": true,
  "sam_lora_rank": 8,
  "sam_lora_alpha": 16.0,
  "sam_lora_dropout": 0.0,
  "command": "python -m seg-rl.heatmap.train --jsonl datasets/... --use_sam_encoder --sam_lora_enabled ...",
  "timestamp": "2025-11-09T10:30:45.123456"
}
```

### 2. 推理参数记录 (`predict_point_sequence_with_sam.py`)

#### 自动保存位置
```
<sam_masks_dir>/inference_args.json          # 推理参数
<sam_masks_dir>/model_training_args.json    # 模型训练参数（自动拷贝）
```

#### 保存内容

**inference_args.json**:
- ✅ 所有推理参数
- ✅ 完整的执行命令
- ✅ 时间戳

**model_training_args.json** (自动拷贝):
- ✅ 从模型所在目录自动拷贝
- ✅ 记录模型的训练配置
- ✅ 完整的训练-推理追溯链

#### 示例输出

**inference_args.json**:
```json
{
  "model_path": "outputs/braintumour/heatmap_train-latelora-251109/model_epoch_15.pt",
  "images_dir": "datasets/seg_r1_md/Task01_BrainTumour/canonical/images",
  "masks_dir": "datasets/seg_r1_md/Task01_BrainTumour/canonical/masks",
  "output_jsonl": "outputs/braintumour/pred_supervised-r8-epoch15-251109.jsonl",
  "sam_masks_dir": "outputs/braintumour/pred_supervised-r8-epoch15-251109",
  "sam_checkpoint": "third_party/sam2/checkpoints/sam2.1_hiera_large.pt",
  "num_points": 17,
  "device": "cuda",
  "resize": [512, 512],
  "height": 512,
  "width": 512,
  "command": "python seg-rl/heatmap/predict_point_sequence_with_sam.py --model_path ... --num_points 17 ...",
  "timestamp": "2025-11-09T14:20:30.789012"
}
```

**model_training_args.json** (自动拷贝):
```json
{
  "use_sam_encoder": true,
  "sam_lora_enabled": true,
  "sam_lora_rank": 8,
  "sam_lora_alpha": 16.0,
  "epochs": 100,
  "lr": 0.0001,
  ...
}
```

## 使用场景

### 场景 1: 查看训练配置

```bash
# 查看某个模型的训练参数
cat outputs/braintumour/heatmap_train-latelora-251109/training_args.json

# 或使用 jq 格式化查看
jq . outputs/braintumour/heatmap_train-latelora-251109/training_args.json
```

### 场景 2: 复现实验

从 JSON 文件中获取完整命令：

```bash
# 提取命令
jq -r '.command' outputs/braintumour/heatmap_train-latelora-251109/training_args.json

# 直接运行（bash）
eval $(jq -r '.command' outputs/braintumour/heatmap_train-latelora-251109/training_args.json)
```

### 场景 3: 追溯推理结果

当您有一个推理结果目录时，可以完整追溯：

```bash
# 1. 查看推理参数
cat outputs/braintumour/pred_supervised-r8-epoch15-251109/inference_args.json

# 2. 查看使用的模型训练参数
cat outputs/braintumour/pred_supervised-r8-epoch15-251109/model_training_args.json

# 3. 完整的实验链路追溯
训练配置 → 训练模型 → 推理配置 → 推理结果
```

### 场景 4: 对比实验

```bash
# 对比两个实验的配置差异
diff \
  <(jq -S . outputs/exp1/training_args.json) \
  <(jq -S . outputs/exp2/training_args.json)
```

### 场景 5: 批量实验分析

```python
import json
import glob

# 收集所有实验的配置
experiments = []
for path in glob.glob("outputs/**/training_args.json", recursive=True):
    with open(path) as f:
        config = json.load(f)
        config['output_dir'] = str(path).replace('/training_args.json', '')
        experiments.append(config)

# 分析哪些参数组合效果最好
import pandas as pd
df = pd.DataFrame(experiments)
print(df[['output_dir', 'sam_lora_rank', 'sam_lora_alpha', 'lr', 'epochs']])
```

## 文件结构示例

### 训练输出目录

```
outputs/braintumour/heatmap_train-latelora-251109/
├── training_args.json          # ← 训练参数（新增）
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

### 推理输出目录

```
outputs/braintumour/pred_supervised-r8-epoch15-251109/
├── inference_args.json         # ← 推理参数（新增）
├── model_training_args.json    # ← 模型训练参数（自动拷贝，新增）
├── BRATS_001_z0029/
│   ├── 0.png
│   ├── 1.png
│   └── ...
├── BRATS_001_z0030/
│   └── ...
└── ...
```

## 优势

### 1. 📝 **完整记录**
- 每次实验的所有参数都被记录
- 不再需要手动记录或从日志中提取

### 2. 🔄 **易于复现**
- JSON 包含完整命令，可直接重新运行
- 参数清晰，易于理解和修改

### 3. 🔍 **可追溯性**
- 推理结果可以追溯到训练配置
- 完整的实验链路：数据 → 训练 → 推理 → 结果

### 4. 📊 **批量分析**
- JSON 格式易于程序化处理
- 可以批量分析多个实验的参数和结果

### 5. 🤝 **团队协作**
- 共享实验配置更容易
- 其他人可以准确复现您的实验

## 技术细节

### 参数序列化

使用 Python 的 `vars(args)` 将 argparse Namespace 转换为字典：

```python
args_dict = vars(args).copy()
args_dict['command'] = ' '.join(sys.argv)  # 完整命令
args_dict['timestamp'] = datetime.now().isoformat()  # ISO 8601 格式时间戳
```

### 文件命名规范

- `training_args.json` - 训练脚本生成
- `inference_args.json` - 推理脚本生成
- `model_training_args.json` - 从模型目录拷贝的训练参数

### 自动拷贝逻辑

```python
# 推理时自动查找并拷贝训练参数
model_dir = os.path.dirname(args.model_path)
training_args_src = os.path.join(model_dir, "training_args.json")
if os.path.isfile(training_args_src):
    training_args_dst = os.path.join(args.sam_masks_dir, "model_training_args.json")
    shutil.copy2(training_args_src, training_args_dst)
```

## 示例工作流

### 完整实验流程

```bash
# 1. 训练模型
python -m seg-rl.heatmap.train \
  --jsonl data.jsonl \
  --sam_dir masks/ \
  --use_sam_encoder \
  --sam_checkpoint sam2.pt \
  --sam_lora_enabled \
  --sam_lora_rank 8 \
  --epochs 100 \
  --out_dir outputs/exp001

# → 自动生成: outputs/exp001/training_args.json

# 2. 进行推理
python seg-rl/heatmap/predict_point_sequence_with_sam.py \
  --model_path outputs/exp001/model_epoch_100.pt \
  --images_dir test_images/ \
  --masks_dir test_masks/ \
  --output_jsonl outputs/exp001_pred/results.jsonl \
  --sam_masks_dir outputs/exp001_pred \
  --sam_checkpoint sam2.pt \
  --num_points 17

# → 自动生成: 
#   - outputs/exp001_pred/inference_args.json
#   - outputs/exp001_pred/model_training_args.json (从 outputs/exp001/ 拷贝)

# 3. 查看完整配置链
echo "=== 训练配置 ==="
cat outputs/exp001/training_args.json | jq .

echo "=== 推理配置 ==="
cat outputs/exp001_pred/inference_args.json | jq .

echo "=== 模型训练配置（推理时拷贝）==="
cat outputs/exp001_pred/model_training_args.json | jq .
```

### 快速查询关键参数

```bash
# 查看 LoRA 配置
jq '{sam_lora_enabled, sam_lora_rank, sam_lora_alpha, sam_lora_dropout}' \
  outputs/exp001/training_args.json

# 查看学习率设置
jq '{lr, sam_lora_lr, lr_scheduler, warmup_epochs}' \
  outputs/exp001/training_args.json

# 查看数据配置
jq '{jsonl, sam_dir, height, width, batch_size}' \
  outputs/exp001/training_args.json
```

## 注意事项

### 1. 文件覆盖

每次运行都会覆盖之前的参数文件。如果使用 `--auto_resume`，参数文件会被更新为最新的运行参数。

### 2. 敏感信息

参数文件包含完整路径。如果需要分享，可能需要脱敏处理：

```python
import json

with open('training_args.json') as f:
    args = json.load(f)

# 移除敏感路径
for key in ['jsonl', 'sam_dir', 'sam_checkpoint', 'out_dir']:
    if key in args:
        args[key] = args[key].replace('/root/autodl-tmp/works/', '<workspace>/')

with open('training_args_clean.json', 'w') as f:
    json.dump(args, f, indent=2)
```

### 3. 版本控制

建议将参数文件纳入版本控制：

```bash
# .gitignore
outputs/**/*.pt          # 不提交模型文件
outputs/**/*.png         # 不提交图片
!outputs/**/training_args.json      # 但提交参数文件
!outputs/**/inference_args.json
```

## 工具脚本

### 查看实验参数的工具

创建 `tools/show_exp_config.py`:

```python
#!/usr/bin/env python3
"""查看实验配置的工具脚本"""

import json
import sys
from pathlib import Path

def show_config(path: str):
    """显示配置文件内容"""
    with open(path) as f:
        config = json.load(f)
    
    print(f"\n{'='*60}")
    print(f"配置文件: {path}")
    print(f"{'='*60}")
    
    # 基本信息
    print(f"\n时间戳: {config.get('timestamp', 'N/A')}")
    
    # 关键参数
    print("\n关键参数:")
    keys = ['use_sam_encoder', 'sam_lora_enabled', 'sam_lora_rank', 
            'sam_lora_alpha', 'epochs', 'lr', 'batch_size', 'loss']
    for k in keys:
        if k in config:
            print(f"  {k}: {config[k]}")
    
    # 完整命令
    print(f"\n完整命令:")
    print(f"  {config.get('command', 'N/A')}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python show_exp_config.py <path_to_args.json>")
        sys.exit(1)
    
    show_config(sys.argv[1])
```

### 对比两个实验

创建 `tools/compare_exp_configs.py`:

```python
#!/usr/bin/env python3
"""对比两个实验配置的差异"""

import json
import sys

def compare_configs(path1: str, path2: str):
    """对比两个配置文件"""
    with open(path1) as f:
        config1 = json.load(f)
    with open(path2) as f:
        config2 = json.load(f)
    
    # 找出差异
    all_keys = set(config1.keys()) | set(config2.keys())
    differences = {}
    
    for key in sorted(all_keys):
        val1 = config1.get(key)
        val2 = config2.get(key)
        if val1 != val2:
            differences[key] = {'exp1': val1, 'exp2': val2}
    
    print(f"\n{'='*60}")
    print(f"配置对比")
    print(f"{'='*60}")
    print(f"实验 1: {path1}")
    print(f"实验 2: {path2}")
    print(f"\n发现 {len(differences)} 处差异:\n")
    
    for key, vals in differences.items():
        print(f"{key}:")
        print(f"  实验 1: {vals['exp1']}")
        print(f"  实验 2: {vals['exp2']}")
        print()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python compare_exp_configs.py <config1.json> <config2.json>")
        sys.exit(1)
    
    compare_configs(sys.argv[1], sys.argv[2])
```

## 实际示例

### 训练后查看配置

```bash
# 训练完成后
ls -lh outputs/braintumour/heatmap_train-latelora-251109/

# 查看保存的参数
cat outputs/braintumour/heatmap_train-latelora-251109/training_args.json | jq '.'

# 提取关键信息
echo "LoRA Rank: $(jq -r '.sam_lora_rank' outputs/braintumour/heatmap_train-latelora-251109/training_args.json)"
echo "Learning Rate: $(jq -r '.lr' outputs/braintumour/heatmap_train-latelora-251109/training_args.json)"
echo "Epochs: $(jq -r '.epochs' outputs/braintumour/heatmap_train-latelora-251109/training_args.json)"
```

### 推理后查看完整链路

```bash
# 推理完成后
ls -lh outputs/braintumour/pred_supervised-r8-epoch15-251109/

# 文件列表：
# - inference_args.json          ← 推理参数
# - model_training_args.json     ← 模型训练参数（拷贝）
# - BRATS_*/                     ← 预测结果

# 查看推理配置
jq '{model_path, num_points, resize, timestamp}' \
  outputs/braintumour/pred_supervised-r8-epoch15-251109/inference_args.json

# 查看模型训练配置
jq '{sam_lora_enabled, sam_lora_rank, epochs, lr}' \
  outputs/braintumour/pred_supervised-r8-epoch15-251109/model_training_args.json
```

## 最佳实践

### 1. 命名规范

建议使用描述性的输出目录名：

```bash
# 包含关键参数的目录名
--out_dir outputs/braintumour/train_r8_alpha16_e100_20251109
--sam_masks_dir outputs/braintumour/pred_r8_e100_n17_20251109
```

### 2. 实验日志

结合参数文件和训练日志：

```bash
python -m seg-rl.heatmap.train \
  --out_dir outputs/exp001 \
  ... 其他参数 ... \
  2>&1 | tee outputs/exp001/train.log

# 现在您有：
# - training_args.json  ← 结构化参数
# - train.log           ← 完整日志
```

### 3. README 文件

在输出目录创建 README：

```bash
cd outputs/braintumour/heatmap_train-latelora-251109/

cat > README.md << 'EOF'
# 实验说明

## 训练配置
见 `training_args.json`

## 模型检查点
- model_epoch_5.pt
- model_epoch_10.pt
- model_epoch_15.pt (最佳)
- last.pt

## 结果
- Val PCK@10: 0.85
- Test PCK@10: 0.83

## 备注
使用 SAM Late LoRA (rank=8) 进行训练
EOF
```

## 迁移说明

### 旧实验目录

如果您有旧的实验目录（没有 `training_args.json`），可以手动创建：

```python
# 手动创建参数文件
import json

config = {
    "note": "Manually created for old experiment",
    "jsonl": "path/to/data.jsonl",
    "epochs": 100,
    "lr": 0.0001,
    # ... 其他已知参数
}

with open('outputs/old_exp/training_args.json', 'w') as f:
    json.dump(config, f, indent=2)
```

## 总结

现在每次运行训练或推理脚本时：

✅ **训练**: 自动保存 `training_args.json` 到输出目录  
✅ **推理**: 自动保存 `inference_args.json` 和拷贝 `model_training_args.json`  
✅ **追溯**: 完整的参数链路记录  
✅ **复现**: 一键获取运行命令  
✅ **分析**: 方便批量对比实验  

这大大提升了实验管理的效率和可维护性！🎉

