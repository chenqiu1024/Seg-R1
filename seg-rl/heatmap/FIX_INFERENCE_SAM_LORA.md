# SAM Late LoRA 推理问题修复

## 问题描述

使用训练好的 SAM Late LoRA 模型进行推理时，所有预测的提示点都集中在 [119.x, 119.x] 附近，这表明模型没有正确加载。

## 根本原因

### 训练与推理的模型不匹配

1. **训练时**：使用 `PointHeatmapModelWithSAM`
   - 包含 SAM image encoder
   - 包含 Late LoRA 适配器
   - Checkpoint 中保存了 `sam_encoder.*` 参数

2. **推理时**：`predict_next_point_from_model.py` 只加载 `PointHeatmapModel`
   - 标准模型，不包含 SAM encoder
   - 模型结构不匹配
   - `load_state_dict(strict=False)` 导致所有 SAM 参数被忽略
   - 使用随机初始化的权重进行预测

### 具体错误流程

```python
# 训练保存的 checkpoint 结构：
{
    "model": {
        "sam_encoder.sam_model.image_encoder...": ...,  # SAM encoder 参数
        "sam_encoder.lora_modules...": ...,              # LoRA 参数
        "cond_encoder...": ...,                          # 其他参数
        ...
    },
    "metadata": {
        "use_sam_encoder": True,
        "sam_lora_enabled": True,
        ...
    }
}

# 推理时加载的模型：
model = PointHeatmapModel(cfg)  # ❌ 错误！应该用 PointHeatmapModelWithSAM
# 结果：所有 sam_encoder.* 参数无法匹配，被忽略
```

## 修复方案

### 1. 修改 `predict_next_point_from_model.py`

#### a) 导入必要的类

```python
from .model import (
    ModelConfig, 
    PointHeatmapModel, 
    PointHeatmapModelWithSAM,  # 新增
    soft_argmax_from_logits
)
from .utils import load_checkpoint  # 新增
```

#### b) 添加 SAM checkpoint 参数

```python
def parse_args():
    # ...
    p.add_argument("--sam_checkpoint", type=str, default=None, 
                   help="Path to SAM checkpoint (required if model uses SAM encoder)")
```

#### c) 重写 `_load_model` 函数

```python
def _load_model(model_path: str, device: torch.device, sam_checkpoint: Optional[str] = None):
    """Load model with automatic type detection from checkpoint metadata"""
    
    # 加载 checkpoint 并检查元数据
    ckpt = torch.load(model_path, map_location="cpu")
    metadata = ckpt.get("metadata", {})
    use_sam_encoder = metadata.get("use_sam_encoder", False)
    sam_lora_enabled = metadata.get("sam_lora_enabled", False)
    
    if use_sam_encoder:
        # 创建 SAM-based 模型
        cfg = ModelConfig(
            backbone="unet_s",
            use_sam_encoder=True,
            sam_checkpoint=sam_checkpoint,
            sam_lora_enabled=sam_lora_enabled,
            sam_lora_rank=metadata.get("sam_lora_rank", 8),
            sam_lora_alpha=metadata.get("sam_lora_alpha", 16.0),
            ...
        )
        model = PointHeatmapModelWithSAM(cfg)
    else:
        # 创建标准模型
        model = PointHeatmapModel(cfg)
    
    # 使用 load_checkpoint 正确加载参数
    load_checkpoint(model_path, model, strict=False)
    return model
```

#### d) 更新函数调用

```python
# run_initial 和 run_append 中：
model = _load_model(args.model_path, device, args.sam_checkpoint)
```

### 2. 修改 `predict_point_sequence_with_sam.py`

在 `build_predict_cmd` 函数中传递 SAM checkpoint：

```python
def build_predict_cmd(args: argparse.Namespace, is_first: bool) -> list:
    # ...
    # SAM checkpoint（如果模型使用 SAM encoder）
    if args.sam_checkpoint:
        cmd.extend(["--sam_checkpoint", args.sam_checkpoint])
    
    return cmd
```

## 修复效果

### 修复前

```python
# 所有预测点都集中在固定位置
[119.47419738769531, 119.50137329101562]
[119.39747619628906, 119.50483703613281]
[119.4162826538086, 119.50761413574219]
...
```

### 修复后

```python
# 预测点分布正常
[126.31128787994385, 129.74581718444824]
[127.87633895874023, 157.52596378326416]
[130.29398918151855, 155.44143676757812]
[133.0239486694336, 155.8964967727661]
...
```

## 正确的使用方法

### 使用 SAM Late LoRA 模型推理

```bash
python seg-rl/heatmap/predict_point_sequence_with_sam.py \
  --model_path outputs/braintumour/heatmap_train-latelora-251109/model_epoch_15.pt \
  --images_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/images \
  --masks_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/masks \
  --output_jsonl outputs/braintumour/pred_supervised-r8-epoch15-251109.jsonl \
  --sam_masks_dir outputs/braintumour/pred_supervised-r8-epoch15-251109 \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --device cuda \
  --resize 512 512 \
  --num_points 17 \
  --height 512 \
  --width 512
```

**重要**：必须提供 `--sam_checkpoint` 参数！

### 使用标准模型推理

```bash
python seg-rl/heatmap/predict_point_sequence_with_sam.py \
  --model_path outputs/braintumour/standard_model.pt \
  --images_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/images \
  --masks_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/masks \
  --output_jsonl outputs/braintumour/pred_standard.jsonl \
  --sam_masks_dir outputs/braintumour/pred_standard_masks \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --device cuda \
  --resize 512 512 \
  --num_points 17
```

注意：`--sam_checkpoint` 对标准模型也是必需的（用于 SAM 分割，不是模型推理）

## 自动检测机制

修复后的代码会自动：

1. ✅ 从 checkpoint 读取 `metadata`
2. ✅ 检测是否使用 SAM encoder
3. ✅ 创建正确类型的模型
4. ✅ 正确加载所有参数（包括 LoRA）
5. ✅ 给出清晰的日志输出

### 日志示例

```
[Load Model] Checkpoint type: use_sam_encoder=True, sam_lora_enabled=True
[Load Model] Using SAM checkpoint: third_party/sam2/checkpoints/sam2.1_hiera_large.pt
[SAM] Loading SAM checkpoint from third_party/sam2/checkpoints/sam2.1_hiera_large.pt
[SAM] Enabling Late LoRA: rank=8, alpha=16.0
[Late LoRA] Found 48 blocks in trunk, applying to last block
[Late LoRA] Found attention module: attn
[Late LoRA] Applied to image_encoder.trunk.blocks[-1].attn.qkv: in=1152, out=3456, rank=8, device=cuda:0
[Late LoRA] Applied to image_encoder.trunk.blocks[-1].attn.proj: in=1152, out=1152, rank=8, device=cuda:0

============================================================
LoRA Configuration Summary
============================================================
Total parameters:      224,501,938
Trainable parameters:       55,296
LoRA parameters:           110,592
Trainable ratio:             0.02%
============================================================

[SAM] Encoder feature dimension: 256
[Load Model] Loading checkpoint parameters...
[Checkpoint] Loaded model parameters (strict=False)
[Load Model] Model loaded successfully
```

## 经验教训

1. **训练和推理必须使用相同的模型结构**
2. **使用 `strict=False` 时要小心**：可能会静默忽略参数
3. **在 checkpoint 中保存元数据非常重要**：实现自动兼容性检测
4. **推理脚本应该读取并尊重 checkpoint 元数据**
5. **添加详细日志有助于调试**

## 修改文件列表

1. ✅ `seg-rl/heatmap/predict_next_point_from_model.py`
   - 添加 SAM encoder 支持
   - 自动检测模型类型
   - 正确加载 checkpoint

2. ✅ `seg-rl/heatmap/predict_point_sequence_with_sam.py`
   - 传递 SAM checkpoint 参数

## 测试验证

```bash
# 快速测试（只生成第一个点）
python -m seg-rl.heatmap.predict_next_point_from_model \
  --model_path outputs/braintumour/heatmap_train-latelora-251109/model_epoch_15.pt \
  --images_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/images \
  --masks_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/masks \
  --output_json outputs/braintumour/test_pred.jsonl \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --height 512 --width 512

# 检查结果
python -c "
import json
with open('outputs/braintumour/test_pred.jsonl') as f:
    data = json.load(f)
for i, item in enumerate(data[:5]):
    print(f'{i+1}. {item[\"points\"][0]}, label: {item[\"labels\"][0]}')
"
```

预期输出：点坐标应该分布在不同位置，而不是集中在 [119, 119] 附近。

## 完成时间

2025-11-09

## 状态

✅ 已修复并测试通过

