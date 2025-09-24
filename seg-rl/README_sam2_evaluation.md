# SAM2 Automatic Segmentation Evaluation

这个脚本使用SAM2的Automatic模式对图片进行批量分割，并评估其与真值mask的吻合度。

## 功能特点

### 🎯 核心功能
- **SAM2 Automatic分割**: 对指定目录下的图片进行批量自动分割
- **智能匹配**: 通过包围盒重叠筛选候选预测mask
- **多指标评估**: 支持IoU和DICE两种评估指标
- **统计分析**: 计算整个数据集的最好/最差/平均分割效果
- **可视化输出**: 生成分割结果和对比图片

### 📊 评估流程
1. 使用SAM2 Automatic模式分割每张原图，得到多个候选区域
2. 通过包围盒重叠筛选与真值相关的预测mask
3. 计算筛选后的预测mask与真值mask的IoU/DICE
4. 取最高分作为该图片的分割效果评估
5. 统计整个数据集的分割效果分布

## 安装依赖

### 必需依赖
```bash
pip install opencv-python numpy torch pillow
```

### 可选依赖（用于可视化）
```bash
pip install matplotlib
```

### SAM2安装
确保已按照SAM2官方说明安装：
```bash
git clone https://github.com/facebookresearch/segment-anything-2.git third_party/sam2
cd third_party/sam2 && pip install -e .
```

## 使用方法

### 基础用法
```bash
python seg-rl/sam2_automatic_evaluation.py \
  --image_dir /path/to/images \
  --json_file /path/to/masks.json \
  --sam_checkpoint /path/to/sam2.1_hiera_large.pt \
  --output_dir /path/to/results \
  --device cuda
```

### 实际示例
```bash
python seg-rl/sam2_automatic_evaluation.py \
  --image_dir /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/imagesTr \
  --json_file /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/pred_masks-0.json \
  --sam_checkpoint /root/autodl-tmp/works/Seg-R0/third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --output_dir /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/evaluation_results \
  --device cuda \
  --metric dice \
  --verbose
```

### 高级参数调优
```bash
python seg-rl/sam2_automatic_evaluation.py \
  --image_dir /path/to/images \
  --json_file /path/to/masks.json \
  --sam_checkpoint /path/to/sam2.1_hiera_large.pt \
  --output_dir /path/to/results \
  --device cuda \
  --metric iou \
  --points_per_side 64 \
  --pred_iou_thresh 0.9 \
  --stability_score_thresh 0.96 \
  --no_visualization \
  --verbose
```

## 输入数据格式

### 图片目录结构
```
image_dir/
├── image1.jpg
├── image2.png
├── image3.jpg
└── ...
```

### JSON文件格式
真值mask信息的JSON文件格式：
```json
[
  {
    "mask_path": "/path/to/mask1.png",
    "bbox": [10, 20, 100, 150]
  },
  {
    "mask_path": "/path/to/mask2.png", 
    "bbox": [50, 80, 200, 300]
  }
]
```

其中：
- `mask_path`: 真值mask图片的路径
- `bbox`: 包围盒坐标 `[x_min, y_min, x_max, y_max]`

### 文件名对应关系
- 原图文件名：`image1.jpg`
- 对应mask文件名：`image1.png`（仅扩展名可能不同）
- JSON中的记录通过文件名（去扩展名）进行匹配

## 输出结果

### 1. 统计报告
```
Evaluation completed in 120.5 seconds:
  Total images: 100
  Processed: 95
  Errors: 5
  Metric: IOU
  Best score: 0.892
  Worst score: 0.234
  Average score: 0.671
```

### 2. 详细结果JSON
`evaluation_results.json` 包含：
- 整体统计信息
- 每张图片的详细评估结果
- 使用的参数配置

### 3. 可视化图片
对每张处理的图片生成两种可视化：

**图1: `{image_name}_all_masks.png`**
- 显示SAM2 Automatic模式生成的所有分割区域
- 多色显示，每个区域用不同颜色标识
- 包含包围盒边框

**图2: `{image_name}_comparison.png`**
- 灰度原图作为背景
- 绿色半透明：真值mask
- 红色半透明：最佳预测mask
- 包围盒对比
- 评估指标标注

## 参数说明

### 必需参数
- `--image_dir`: 输入图片目录
- `--json_file`: 真值mask信息JSON文件
- `--sam_checkpoint`: SAM2模型检查点路径
- `--output_dir`: 输出结果目录

### 可选参数
- `--device`: 运行设备 (cuda/cpu，默认自动检测)
- `--metric`: 评估指标 (iou/dice，默认iou)
- `--config_path`: SAM2配置文件路径
- `--points_per_side`: 每边点数量 (默认32)
- `--pred_iou_thresh`: IoU阈值 (默认0.88)
- `--stability_score_thresh`: 稳定性分数阈值 (默认0.95)
- `--verbose`: 显示详细处理信息
- `--no_visualization`: 跳过可视化图片生成

## 性能优化建议

### 1. GPU加速
```bash
--device cuda
```

### 2. 调整分割精度
```bash
# 高精度（慢）
--points_per_side 64 --pred_iou_thresh 0.9 --stability_score_thresh 0.96

# 中等精度（平衡）
--points_per_side 32 --pred_iou_thresh 0.88 --stability_score_thresh 0.95

# 快速模式（快）
--points_per_side 16 --pred_iou_thresh 0.8 --stability_score_thresh 0.9
```

### 3. 禁用可视化
```bash
--no_visualization
```

## 故障排除

### 常见问题

1. **SAM2导入失败**
   ```
   Error importing SAM2: No module named 'sam2'
   ```
   解决：确保SAM2正确安装在`third_party/sam2`目录下

2. **CUDA内存不足**
   ```
   RuntimeError: CUDA out of memory
   ```
   解决：使用`--device cpu`或减少`--points_per_side`

3. **matplotlib未安装**
   ```
   Warning: matplotlib not found. Visualization features will be disabled.
   ```
   解决：安装matplotlib或使用`--no_visualization`

4. **文件名不匹配**
   ```
   No ground truth found, skipping
   ```
   解决：确保图片文件名与JSON中的mask文件名对应

### 测试脚本
运行测试脚本检查环境：
```bash
python seg-rl/test_evaluation.py
```

## 示例工作流

1. **准备数据**
   ```bash
   # 确保有原图目录和真值JSON
   ls /path/to/images/
   cat /path/to/masks.json
   ```

2. **运行评估**
   ```bash
   python seg-rl/sam2_automatic_evaluation.py \
     --image_dir /path/to/images \
     --json_file /path/to/masks.json \
     --sam_checkpoint /path/to/sam2.1_hiera_large.pt \
     --output_dir /path/to/results \
     --device cuda \
     --verbose
   ```

3. **查看结果**
   ```bash
   # 查看统计结果
   cat /path/to/results/evaluation_results.json
   
   # 查看可视化结果
   ls /path/to/results/*.png
   ```

## 扩展和定制

脚本设计为模块化，可以轻松扩展：
- 添加新的评估指标
- 自定义可视化样式
- 调整包围盒重叠判断逻辑
- 集成其他分割模型

---

**注意**: 首次运行时SAM2会下载模型权重，可能需要一些时间。建议在稳定的网络环境下进行。
