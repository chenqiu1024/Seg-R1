# SAM2图像分割工具

本目录包含使用SAM2 (Segment Anything Model 2) 从点提示进行图像分割的工具。

## 脚本说明

### 1. `sam2_segment_from_points.py` (独立版本)
- 完全独立的SAM2分割脚本
- 自行实现SAMWrapper类
- 更容易部署和调试

### 2. `sam2_segment_simple.py` (复用版本)  
- 复用`seg-r1/src/open_r1/grpo.py`中的SAMWrapper类
- 代码更简洁，与现有代码保持一致
- 推荐用于项目内部使用

## 安装依赖

### 1. 安装SAM2
```bash
# 克隆SAM2仓库
git clone https://github.com/facebookresearch/segment-anything-2.git third_party/sam2
cd third_party/sam2

# 安装SAM2
pip install -e .

# 创建模型目录
mkdir -p checkpoints
```

### 2. 下载模型权重
```bash
# SAM2.1 Hiera Large模型 (~900MB)
wget -O third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# 或者其他模型大小:
# SAM2.1 Hiera Base+ (~152MB)
# wget -O third_party/sam2/checkpoints/sam2.1_hiera_base_plus.pt \
#   https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
```

### 3. 安装其他依赖
```bash
pip install opencv-python pillow numpy torch
```

## 数据格式

### 输入JSONL格式

**新格式** (推荐):
```json
{"image": "/path/to/image.jpg", "points": [[x1,y1], [x2,y2]], "labels": [1, 0]}
```

**旧格式** (向后兼容):
```json
{"image": "/path/to/image.jpg", "x": x1, "y": y1}
```

- `points`: 点坐标列表 `[[x,y], ...]`
- `labels`: 对应标签列表 `[1, 0, ...]`
  - `1`: 前景点 (正类)
  - `0`: 背景点 (负类)
- `x`, `y`: 旧格式中的单个坐标 (默认为前景点)

### 输出格式
- 灰度PNG图像
- `0` (黑色): 背景
- `255` (白色): 前景
- 文件名与输入图像对应 (仅扩展名改为`.png`)

## 使用示例

### 基础用法
```bash
python seg-rl/sam2_segment_from_points.py \
  --input_jsonl /path/to/training_data.jsonl \
  --output_dir /path/to/output_masks \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --device cuda
```

### 高级选项
```bash
python seg-rl/sam2_segment_from_points.py \
  --input_jsonl /path/to/training_data.jsonl \
  --output_dir /path/to/output_masks \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --device cuda \
  --resize 512 512 \
  --skip_existing
```

### 使用简化版本
```bash
python seg-rl/sam2_segment_simple.py \
  --input_jsonl /path/to/training_data.jsonl \
  --output_dir /path/to/output_masks \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --device cuda
```

## 完整工作流程

### 1. 从mask生成点标注
```bash
python seg-rl/annotator/gen_point_jsonl_from_masks.py \
  --images_dir /path/to/images \
  --masks_dir /path/to/masks \
  --output_jsonl /path/to/points.jsonl
```

### 2. 使用SAM2分割
```bash
python seg-rl/sam2_segment_from_points.py \
  --input_jsonl /path/to/points.jsonl \
  --output_dir /path/to/sam2_masks \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --device cuda
```

### 3. 训练热力图模型
```bash
python -m seg_rl.heatmap.train \
  --data_jsonl /path/to/points.jsonl \
  --arch unet_s \
  --loss kl --sigma 6.0 \
  --epochs 40 --amp \
  --out_dir /path/to/heatmap_model
```

## 参数说明

### 必需参数
- `--input_jsonl`: 输入JSONL文件路径
- `--output_dir`: 输出mask目录
- `--sam_checkpoint`: SAM2模型权重路径

### 可选参数
- `--device`: 运行设备 (`cuda`/`cpu`)，默认自动检测
- `--resize WIDTH HEIGHT`: 调整输入图像大小
- `--skip_existing`: 跳过已存在的输出文件
- `--config_path`: SAM2配置文件路径 (通常不需要修改)

## 性能考虑

### 内存使用
- SAM2 Large模型约需要8GB GPU内存
- 可考虑使用Base+模型减少内存占用
- 大图像可使用`--resize`减小输入尺寸

### 速度优化
- 使用GPU (`--device cuda`) 可显著提升速度
- 批处理大量图像时使用`--skip_existing`避免重复计算
- 考虑并行处理多个JSONL文件

## 故障排除

### 常见错误

1. **SAM2导入失败**
   ```
   Error importing SAM2: No module named 'sam2'
   ```
   - 确保已正确安装SAM2: `cd third_party/sam2 && pip install -e .`

2. **模型文件未找到**
   ```
   Error: SAM2 checkpoint not found
   ```
   - 检查模型路径是否正确
   - 确保已下载模型权重文件

3. **配置文件未找到**
   ```
   SAM2 config file not found: configs/sam2.1/sam2.1_hiera_l.yaml
   ```
   - 确保在正确的目录运行脚本
   - 或使用`--config_path`指定完整路径

4. **GPU内存不足**
   ```
   CUDA out of memory
   ```
   - 使用`--device cpu`切换到CPU
   - 或使用`--resize`减小图像尺寸
   - 考虑使用更小的模型

### 调试建议
- 先用少量样本测试
- 检查输入图像路径是否正确
- 验证点坐标是否在图像范围内
- 使用`--device cpu`排除GPU相关问题

## 扩展功能

可以根据需要扩展脚本功能：

1. **批处理优化**: 实现批量预测以提升效率
2. **多尺度分割**: 支持多个分辨率的分割
3. **后处理**: 添加形态学操作改善mask质量
4. **可视化**: 生成带有点标注和分割结果的可视化图像
5. **评估指标**: 计算与真值mask的IoU等指标

## 相关文件

- `annotator/gen_point_jsonl_from_masks.py`: 从mask生成点标注
- `heatmap/`: 热力图训练和推理模块
- `seg-r1/src/open_r1/grpo.py`: 原始SAMWrapper实现
