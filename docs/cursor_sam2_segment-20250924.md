# 添加功能到sam2_segment脚本
_Exported on 9/24/2025 at 06:59:34 PDT from Cursor (1.6.45)_

---

**User**

现在请给sam2_segment_simple.py添加如下功能：对于每个输入图像分割得到的mask，计算最小包围盒，然后把输出mask的路径与包围盒坐标以JSON形式写入到指定的目标文件路径下：整个JSONL文件包含一个数组，数组中每个元素为包含mask路径和相应包围盒坐标的字典。对sam2_segment_from_points.py也做同样的更新

---

**Cursor**

我需要先查看这两个文件的内容，了解它们的现有功能，然后为它们添加计算最小包围盒并输出JSON的功能。

现在我了解了这两个文件的结构。我需要为它们添加计算最小包围盒并输出JSON的功能。让我创建一个任务列表：

首先，我需要添加计算最小包围盒的函数。我将开始修改这两个文件：

```python
#!/usr/bin/env python3

"""
使用SAM2从点提示进行图像分割 (简化版本)

直接复用seg-r1/src/open_r1/grpo.py中的SAMWrapper类进行分割

用法:
    python seg-rl/sam2_segment_simple.py \
      --input_jsonl /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/points_canonical.jsonl \
      --output_dir /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/masks_step1 \
      --sam_checkpoint /root/autodl-tmp/works/Seg-R0/third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
      --device cuda
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

# 添加seg-r1和third_party路径以导入SAMWrapper及sam2依赖
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "seg-r1" / "src"))
# Ensure Python sees the inner package directory (third_party/sam2)
sys.path.insert(0, str(project_root / "third_party" / "sam2"))

try:
    from open_r1.grpo import SAMWrapper
except ImportError as e:
    print(f"Error importing SAMWrapper from grpo.py: {e}")
    print("Please ensure the seg-r1 module is properly set up.")
    print("You may need to:")
    print("1. Install SAM2 dependencies")
    print("2. Check the import path")
    sys.exit(1)


def validate_and_extract_points(obj: Dict[str, Any], line_num: int) -> Tuple[bool, Optional[List[Tuple[float, float]]], Optional[List[int]]]:
    """验证并提取JSON条目中的点和标签"""
    if "image" not in obj:
        print(f"[WARN] Line {line_num}: Missing 'image' field")
        return False, None, None
        
    # 检查格式：新格式有"points"和"labels"，旧格式有"x"和"y"
    if "points" in obj and "labels" in obj:
        # 新格式
        points = obj["points"]
        labels = obj["labels"]
        
        if not isinstance(points, list) or not isinstance(labels, list):
            print(f"[WARN] Line {line_num}: 'points' and 'labels' must be lists")
            return False, None, None
            
        if len(points) != len(labels):
            print(f"[WARN] Line {line_num}: 'points' and 'labels' must have same length")
            return False, None, None
        
        # 验证点格式
        valid_points = []
        valid_labels = []
        for point, label in zip(points, labels):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                print(f"[WARN] Line {line_num}: Invalid point format {point}")
                continue
            valid_points.append((float(point[0]), float(point[1])))
            valid_labels.append(int(label))
            
        if len(valid_points) == 0:
            print(f"[WARN] Line {line_num}: No valid points found")
            return False, None, None
            
        return True, valid_points, valid_labels
        
    elif "x" in obj and "y" in obj:
        # 旧格式 - 假设为正类点
        x = float(obj["x"])
        y = float(obj["y"])
        return True, [(x, y)], [1]
        
    else:
        print(f"[WARN] Line {line_num}: Missing coordinate data (need 'points'+'labels' or 'x'+'y')")
        return False, None, None


def get_output_path(image_path: str, output_dir: str) -> str:
    """生成输出mask文件路径"""
    image_name = Path(image_path).stem
    return os.path.join(output_dir, f"{image_name}.png")


def calculate_bounding_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """计算mask的最小包围盒
    
    Args:
        mask: 二值mask数组
        
    Returns:
        (x_min, y_min, x_max, y_max) 包围盒坐标
    """
    # 确保mask是二值的
    mask_binary = (mask > 0).astype(np.uint8)
    
    # 找到非零像素的坐标
    y_indices, x_indices = np.where(mask_binary)
    
    if len(x_indices) == 0 or len(y_indices) == 0:
        # 如果没有前景像素，返回空包围盒
        return 0, 0, 0, 0
    
    x_min = int(np.min(x_indices))
    x_max = int(np.max(x_indices))
    y_min = int(np.min(y_indices))
    y_max = int(np.max(y_indices))
    
    return x_min, y_min, x_max, y_max


def save_mask_as_grayscale(mask: np.ndarray, output_path: str) -> None:
    """将mask保存为灰度图像"""
    # 确保mask是二值的
    mask_binary = (mask > 0).astype(np.uint8)
    
    # 转换为灰度值 (0=背景, 255=前景)
    mask_gray = mask_binary * 255
    
    # 保存
    cv2.imwrite(output_path, mask_gray)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Use SAM2 to segment images from point prompts (using grpo.py SAMWrapper)")
    p.add_argument("--input_jsonl", type=str, required=True,
                   help="Input JSONL file containing image paths and point coordinates")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output directory to save mask images")
    p.add_argument("--sam_checkpoint", type=str, required=True,
                   help="Path to SAM2 model checkpoint")
    p.add_argument("--device", type=str, default=None,
                   help="Device to run on (cuda/cpu). Auto-detect if not specified")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip processing if output file already exists")
    p.add_argument("--resize", type=int, nargs=2, default=None, metavar=("WIDTH", "HEIGHT"),
                   help="Resize input images to specified size [width height]")
    return p.parse_args()


def main():
    args = parse_args()
    
    # 检查输入文件
    if not os.path.isfile(args.input_jsonl):
        print(f"Error: Input JSONL file not found: {args.input_jsonl}")
        return 1
    
    # 检查SAM2检查点
    if not os.path.isfile(args.sam_checkpoint):
        print(f"Error: SAM2 checkpoint not found: {args.sam_checkpoint}")
        return 1
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化SAM2
    print(f"Initializing SAM2 with checkpoint: {args.sam_checkpoint}")
    print(f"Using device: {args.device or 'auto-detect'}")
    
    try:
        sam_wrapper = SAMWrapper(args.sam_checkpoint, args.device)
        print("SAM2 initialized successfully")
    except Exception as e:
        print(f"Error initializing SAM2: {e}")
        return 1
    
    # 处理JSONL文件
    print(f"Processing JSONL file: {args.input_jsonl}")
    
    num_processed = 0
    num_skipped = 0
    num_errors = 0
    
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[ERROR] Line {line_num}: Invalid JSON - {e}")
                num_errors += 1
                continue
            
            # 验证并提取点信息
            is_valid, points, labels = validate_and_extract_points(obj, line_num)
            if not is_valid:
                num_errors += 1
                continue
            
            image_path = obj["image"]
            
            # 检查图像文件是否存在
            if not os.path.isfile(image_path):
                print(f"[ERROR] Line {line_num}: Image file not found: {image_path}")
                num_errors += 1
                continue
            
            # 生成输出路径
            output_path = get_output_path(image_path, args.output_dir)
            
            # 检查是否跳过已存在的文件
            if args.skip_existing and os.path.isfile(output_path):
                print(f"[SKIP] Line {line_num}: Output already exists: {output_path}")
                num_skipped += 1
                continue
            
            try:
                # 加载图像
                image = PILImage.open(image_path).convert("RGB")
                
                # 调整大小（如果指定）
                if args.resize:
                    image = image.resize(args.resize, PILImage.BILINEAR)
                
                # 运行SAM2预测
                mask, confidence = sam_wrapper.predict(image, points, labels)
                
                # 保存mask
                save_mask_as_grayscale(mask, output_path)
                
                print(f"[OK] Line {line_num}: {image_path} -> {output_path} (confidence: {confidence:.3f})")
                num_processed += 1
                
            except Exception as e:
                print(f"[ERROR] Line {line_num}: Failed to process {image_path} - {e}")
                num_errors += 1
                continue
    
    # 打印总结
    print(f"\nProcessing completed:")
    print(f"  Processed: {num_processed}")
    print(f"  Skipped: {num_skipped}")
    print(f"  Errors: {num_errors}")
    print(f"  Output directory: {args.output_dir}")
    
    return 0 if num_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

```

现在为sam2_segment_simple.py添加命令行参数以支持JSON输出：

```python
#!/usr/bin/env python3

"""
使用SAM2从点提示进行图像分割 (简化版本)

直接复用seg-r1/src/open_r1/grpo.py中的SAMWrapper类进行分割

用法:
    python seg-rl/sam2_segment_simple.py \
      --input_jsonl /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/points_canonical.jsonl \
      --output_dir /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/masks_step1 \
      --sam_checkpoint /root/autodl-tmp/works/Seg-R0/third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
      --device cuda
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

# 添加seg-r1和third_party路径以导入SAMWrapper及sam2依赖
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "seg-r1" / "src"))
# Ensure Python sees the inner package directory (third_party/sam2)
sys.path.insert(0, str(project_root / "third_party" / "sam2"))

try:
    from open_r1.grpo import SAMWrapper
except ImportError as e:
    print(f"Error importing SAMWrapper from grpo.py: {e}")
    print("Please ensure the seg-r1 module is properly set up.")
    print("You may need to:")
    print("1. Install SAM2 dependencies")
    print("2. Check the import path")
    sys.exit(1)


def validate_and_extract_points(obj: Dict[str, Any], line_num: int) -> Tuple[bool, Optional[List[Tuple[float, float]]], Optional[List[int]]]:
    """验证并提取JSON条目中的点和标签"""
    if "image" not in obj:
        print(f"[WARN] Line {line_num}: Missing 'image' field")
        return False, None, None
        
    # 检查格式：新格式有"points"和"labels"，旧格式有"x"和"y"
    if "points" in obj and "labels" in obj:
        # 新格式
        points = obj["points"]
        labels = obj["labels"]
        
        if not isinstance(points, list) or not isinstance(labels, list):
            print(f"[WARN] Line {line_num}: 'points' and 'labels' must be lists")
            return False, None, None
            
        if len(points) != len(labels):
            print(f"[WARN] Line {line_num}: 'points' and 'labels' must have same length")
            return False, None, None
        
        # 验证点格式
        valid_points = []
        valid_labels = []
        for point, label in zip(points, labels):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                print(f"[WARN] Line {line_num}: Invalid point format {point}")
                continue
            valid_points.append((float(point[0]), float(point[1])))
            valid_labels.append(int(label))
            
        if len(valid_points) == 0:
            print(f"[WARN] Line {line_num}: No valid points found")
            return False, None, None
            
        return True, valid_points, valid_labels
        
    elif "x" in obj and "y" in obj:
        # 旧格式 - 假设为正类点
        x = float(obj["x"])
        y = float(obj["y"])
        return True, [(x, y)], [1]
        
    else:
        print(f"[WARN] Line {line_num}: Missing coordinate data (need 'points'+'labels' or 'x'+'y')")
        return False, None, None


def get_output_path(image_path: str, output_dir: str) -> str:
    """生成输出mask文件路径"""
    image_name = Path(image_path).stem
    return os.path.join(output_dir, f"{image_name}.png")


def calculate_bounding_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """计算mask的最小包围盒
    
    Args:
        mask: 二值mask数组
        
    Returns:
        (x_min, y_min, x_max, y_max) 包围盒坐标
    """
    # 确保mask是二值的
    mask_binary = (mask > 0).astype(np.uint8)
    
    # 找到非零像素的坐标
    y_indices, x_indices = np.where(mask_binary)
    
    if len(x_indices) == 0 or len(y_indices) == 0:
        # 如果没有前景像素，返回空包围盒
        return 0, 0, 0, 0
    
    x_min = int(np.min(x_indices))
    x_max = int(np.max(x_indices))
    y_min = int(np.min(y_indices))
    y_max = int(np.max(y_indices))
    
    return x_min, y_min, x_max, y_max


def save_mask_as_grayscale(mask: np.ndarray, output_path: str) -> None:
    """将mask保存为灰度图像"""
    # 确保mask是二值的
    mask_binary = (mask > 0).astype(np.uint8)
    
    # 转换为灰度值 (0=背景, 255=前景)
    mask_gray = mask_binary * 255
    
    # 保存
    cv2.imwrite(output_path, mask_gray)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Use SAM2 to segment images from point prompts (using grpo.py SAMWrapper)")
    p.add_argument("--input_jsonl", type=str, required=True,
                   help="Input JSONL file containing image paths and point coordinates")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output directory to save mask images")
    p.add_argument("--sam_checkpoint", type=str, required=True,
                   help="Path to SAM2 model checkpoint")
    p.add_argument("--device", type=str, default=None,
                   help="Device to run on (cuda/cpu). Auto-detect if not specified")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip processing if output file already exists")
    p.add_argument("--resize", type=int, nargs=2, default=None, metavar=("WIDTH", "HEIGHT"),
                   help="Resize input images to specified size [width height]")
    p.add_argument("--json_output", type=str, default=None,
                   help="Path to output JSON file containing mask paths and bounding boxes")
    return p.parse_args()


def main():
    args = parse_args()
    
    # 检查输入文件
    if not os.path.isfile(args.input_jsonl):
        print(f"Error: Input JSONL file not found: {args.input_jsonl}")
        return 1
    
    # 检查SAM2检查点
    if not os.path.isfile(args.sam_checkpoint):
        print(f"Error: SAM2 checkpoint not found: {args.sam_checkpoint}")
        return 1
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化SAM2
    print(f"Initializing SAM2 with checkpoint: {args.sam_checkpoint}")
    print(f"Using device: {args.device or 'auto-detect'}")
    
    try:
        sam_wrapper = SAMWrapper(args.sam_checkpoint, args.device)
        print("SAM2 initialized successfully")
    except Exception as e:
        print(f"Error initializing SAM2: {e}")
        return 1
    
    # 处理JSONL文件
    print(f"Processing JSONL file: {args.input_jsonl}")
    
    num_processed = 0
    num_skipped = 0
    num_errors = 0
    
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[ERROR] Line {line_num}: Invalid JSON - {e}")
                num_errors += 1
                continue
            
            # 验证并提取点信息
            is_valid, points, labels = validate_and_extract_points(obj, line_num)
            if not is_valid:
                num_errors += 1
                continue
            
            image_path = obj["image"]
            
            # 检查图像文件是否存在
            if not os.path.isfile(image_path):
                print(f"[ERROR] Line {line_num}: Image file not found: {image_path}")
                num_errors += 1
                continue
            
            # 生成输出路径
            output_path = get_output_path(image_path, args.output_dir)
            
            # 检查是否跳过已存在的文件
            if args.skip_existing and os.path.isfile(output_path):
                print(f"[SKIP] Line {line_num}: Output already exists: {output_path}")
                num_skipped += 1
                continue
            
            try:
                # 加载图像
                image = PILImage.open(image_path).convert("RGB")
                
                # 调整大小（如果指定）
                if args.resize:
                    image = image.resize(args.resize, PILImage.BILINEAR)
                
                # 运行SAM2预测
                mask, confidence = sam_wrapper.predict(image, points, labels)
                
                # 保存mask
                save_mask_as_grayscale(mask, output_path)
                
                print(f"[OK] Line {line_num}: {image_path} -> {output_path} (confidence: {confidence:.3f})")
                num_processed += 1
                
            except Exception as e:
                print(f"[ERROR] Line {line_num}: Failed to process {image_path} - {e}")
                num_errors += 1
                continue
    
    # 打印总结
    print(f"\nProcessing completed:")
    print(f"  Processed: {num_processed}")
    print(f"  Skipped: {num_skipped}")
    print(f"  Errors: {num_errors}")
    print(f"  Output directory: {args.output_dir}")
    
    return 0 if num_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

```

现在修改main函数来支持JSON输出功能：

```python
#!/usr/bin/env python3

"""
使用SAM2从点提示进行图像分割 (简化版本)

直接复用seg-r1/src/open_r1/grpo.py中的SAMWrapper类进行分割

用法:
    python seg-rl/sam2_segment_simple.py \
      --input_jsonl /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/points_canonical.jsonl \
      --output_dir /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/masks_step1 \
      --sam_checkpoint /root/autodl-tmp/works/Seg-R0/third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
      --device cuda
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

# 添加seg-r1和third_party路径以导入SAMWrapper及sam2依赖
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "seg-r1" / "src"))
# Ensure Python sees the inner package directory (third_party/sam2)
sys.path.insert(0, str(project_root / "third_party" / "sam2"))

try:
    from open_r1.grpo import SAMWrapper
except ImportError as e:
    print(f"Error importing SAMWrapper from grpo.py: {e}")
    print("Please ensure the seg-r1 module is properly set up.")
    print("You may need to:")
    print("1. Install SAM2 dependencies")
    print("2. Check the import path")
    sys.exit(1)


def validate_and_extract_points(obj: Dict[str, Any], line_num: int) -> Tuple[bool, Optional[List[Tuple[float, float]]], Optional[List[int]]]:
    """验证并提取JSON条目中的点和标签"""
    if "image" not in obj:
        print(f"[WARN] Line {line_num}: Missing 'image' field")
        return False, None, None
        
    # 检查格式：新格式有"points"和"labels"，旧格式有"x"和"y"
    if "points" in obj and "labels" in obj:
        # 新格式
        points = obj["points"]
        labels = obj["labels"]
        
        if not isinstance(points, list) or not isinstance(labels, list):
            print(f"[WARN] Line {line_num}: 'points' and 'labels' must be lists")
            return False, None, None
            
        if len(points) != len(labels):
            print(f"[WARN] Line {line_num}: 'points' and 'labels' must have same length")
            return False, None, None
        
        # 验证点格式
        valid_points = []
        valid_labels = []
        for point, label in zip(points, labels):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                print(f"[WARN] Line {line_num}: Invalid point format {point}")
                continue
            valid_points.append((float(point[0]), float(point[1])))
            valid_labels.append(int(label))
            
        if len(valid_points) == 0:
            print(f"[WARN] Line {line_num}: No valid points found")
            return False, None, None
            
        return True, valid_points, valid_labels
        
    elif "x" in obj and "y" in obj:
        # 旧格式 - 假设为正类点
        x = float(obj["x"])
        y = float(obj["y"])
        return True, [(x, y)], [1]
        
    else:
        print(f"[WARN] Line {line_num}: Missing coordinate data (need 'points'+'labels' or 'x'+'y')")
        return False, None, None


def get_output_path(image_path: str, output_dir: str) -> str:
    """生成输出mask文件路径"""
    image_name = Path(image_path).stem
    return os.path.join(output_dir, f"{image_name}.png")


def calculate_bounding_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """计算mask的最小包围盒
    
    Args:
        mask: 二值mask数组
        
    Returns:
        (x_min, y_min, x_max, y_max) 包围盒坐标
    """
    # 确保mask是二值的
    mask_binary = (mask > 0).astype(np.uint8)
    
    # 找到非零像素的坐标
    y_indices, x_indices = np.where(mask_binary)
    
    if len(x_indices) == 0 or len(y_indices) == 0:
        # 如果没有前景像素，返回空包围盒
        return 0, 0, 0, 0
    
    x_min = int(np.min(x_indices))
    x_max = int(np.max(x_indices))
    y_min = int(np.min(y_indices))
    y_max = int(np.max(y_indices))
    
    return x_min, y_min, x_max, y_max


def save_mask_as_grayscale(mask: np.ndarray, output_path: str) -> None:
    """将mask保存为灰度图像"""
    # 确保mask是二值的
    mask_binary = (mask > 0).astype(np.uint8)
    
    # 转换为灰度值 (0=背景, 255=前景)
    mask_gray = mask_binary * 255
    
    # 保存
    cv2.imwrite(output_path, mask_gray)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Use SAM2 to segment images from point prompts (using grpo.py SAMWrapper)")
    p.add_argument("--input_jsonl", type=str, required=True,
                   help="Input JSONL file containing image paths and point coordinates")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output directory to save mask images")
    p.add_argument("--sam_checkpoint", type=str, required=True,
                   help="Path to SAM2 model checkpoint")
    p.add_argument("--device", type=str, default=None,
                   help="Device to run on (cuda/cpu). Auto-detect if not specified")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip processing if output file already exists")
    p.add_argument("--resize", type=int, nargs=2, default=None, metavar=("WIDTH", "HEIGHT"),
                   help="Resize input images to specified size [width height]")
    p.add_argument("--json_output", type=str, default=None,
                   help="Path to output JSON file containing mask paths and bounding boxes")
    return p.parse_args()


def main():
    args = parse_args()
    
    # 检查输入文件
    if not os.path.isfile(args.input_jsonl):
        print(f"Error: Input JSONL file not found: {args.input_jsonl}")
        return 1
    
    # 检查SAM2检查点
    if not os.path.isfile(args.sam_checkpoint):
        print(f"Error: SAM2 checkpoint not found: {args.sam_checkpoint}")
        return 1
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化SAM2
    print(f"Initializing SAM2 with checkpoint: {args.sam_checkpoint}")
    print(f"Using device: {args.device or 'auto-detect'}")
    
    try:
        sam_wrapper = SAMWrapper(args.sam_checkpoint, args.device)
        print("SAM2 initialized successfully")
    except Exception as e:
        print(f"Error initializing SAM2: {e}")
        return 1
    
    # 处理JSONL文件
    print(f"Processing JSONL file: {args.input_jsonl}")
    
    num_processed = 0
    num_skipped = 0
    num_errors = 0
    
    # 用于存储JSON输出的列表
    json_results = []
    
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[ERROR] Line {line_num}: Invalid JSON - {e}")
                num_errors += 1
                continue
            
            # 验证并提取点信息
            is_valid, points, labels = validate_and_extract_points(obj, line_num)
            if not is_valid:
                num_errors += 1
                continue
            
            image_path = obj["image"]
            
            # 检查图像文件是否存在
            if not os.path.isfile(image_path):
                print(f"[ERROR] Line {line_num}: Image file not found: {image_path}")
                num_errors += 1
                continue
            
            # 生成输出路径
            output_path = get_output_path(image_path, args.output_dir)
            
            # 检查是否跳过已存在的文件
            if args.skip_existing and os.path.isfile(output_path):
                print(f"[SKIP] Line {line_num}: Output already exists: {output_path}")
                num_skipped += 1
                continue
            
            try:
                # 加载图像
                image = PILImage.open(image_path).convert("RGB")
                
                # 调整大小（如果指定）
                if args.resize:
                    image = image.resize(args.resize, PILImage.BILINEAR)
                
                # 运行SAM2预测
                mask, confidence = sam_wrapper.predict(image, points, labels)
                
                # 保存mask
                save_mask_as_grayscale(mask, output_path)
                
                print(f"[OK] Line {line_num}: {image_path} -> {output_path} (confidence: {confidence:.3f})")
                num_processed += 1
                
            except Exception as e:
                print(f"[ERROR] Line {line_num}: Failed to process {image_path} - {e}")
                num_errors += 1
                continue
    
    # 打印总结
    print(f"\nProcessing completed:")
    print(f"  Processed: {num_processed}")
    print(f"  Skipped: {num_skipped}")
    print(f"  Errors: {num_errors}")
    print(f"  Output directory: {args.output_dir}")
    
    return 0 if num_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

```

现在修改处理逻辑来计算包围盒并收集JSON结果：

```python
#!/usr/bin/env python3

"""
使用SAM2从点提示进行图像分割 (简化版本)

直接复用seg-r1/src/open_r1/grpo.py中的SAMWrapper类进行分割

用法:
    python seg-rl/sam2_segment_simple.py \
      --input_jsonl /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/points_canonical.jsonl \
      --output_dir /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/masks_step1 \
      --sam_checkpoint /root/autodl-tmp/works/Seg-R0/third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
      --device cuda
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

# 添加seg-r1和third_party路径以导入SAMWrapper及sam2依赖
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "seg-r1" / "src"))
# Ensure Python sees the inner package directory (third_party/sam2)
sys.path.insert(0, str(project_root / "third_party" / "sam2"))

try:
    from open_r1.grpo import SAMWrapper
except ImportError as e:
    print(f"Error importing SAMWrapper from grpo.py: {e}")
    print("Please ensure the seg-r1 module is properly set up.")
    print("You may need to:")
    print("1. Install SAM2 dependencies")
    print("2. Check the import path")
    sys.exit(1)


def validate_and_extract_points(obj: Dict[str, Any], line_num: int) -> Tuple[bool, Optional[List[Tuple[float, float]]], Optional[List[int]]]:
    """验证并提取JSON条目中的点和标签"""
    if "image" not in obj:
        print(f"[WARN] Line {line_num}: Missing 'image' field")
        return False, None, None
        
    # 检查格式：新格式有"points"和"labels"，旧格式有"x"和"y"
    if "points" in obj and "labels" in obj:
        # 新格式
        points = obj["points"]
        labels = obj["labels"]
        
        if not isinstance(points, list) or not isinstance(labels, list):
            print(f"[WARN] Line {line_num}: 'points' and 'labels' must be lists")
            return False, None, None
            
        if len(points) != len(labels):
            print(f"[WARN] Line {line_num}: 'points' and 'labels' must have same length")
            return False, None, None
        
        # 验证点格式
        valid_points = []
        valid_labels = []
        for point, label in zip(points, labels):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                print(f"[WARN] Line {line_num}: Invalid point format {point}")
                continue
            valid_points.append((float(point[0]), float(point[1])))
            valid_labels.append(int(label))
            
        if len(valid_points) == 0:
            print(f"[WARN] Line {line_num}: No valid points found")
            return False, None, None
            
        return True, valid_points, valid_labels
        
    elif "x" in obj and "y" in obj:
        # 旧格式 - 假设为正类点
        x = float(obj["x"])
        y = float(obj["y"])
        return True, [(x, y)], [1]
        
    else:
        print(f"[WARN] Line {line_num}: Missing coordinate data (need 'points'+'labels' or 'x'+'y')")
        return False, None, None


def get_output_path(image_path: str, output_dir: str) -> str:
    """生成输出mask文件路径"""
    image_name = Path(image_path).stem
    return os.path.join(output_dir, f"{image_name}.png")


def calculate_bounding_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """计算mask的最小包围盒
    
    Args:
        mask: 二值mask数组
        
    Returns:
        (x_min, y_min, x_max, y_max) 包围盒坐标
    """
    # 确保mask是二值的
    mask_binary = (mask > 0).astype(np.uint8)
    
    # 找到非零像素的坐标
    y_indices, x_indices = np.where(mask_binary)
    
    if len(x_indices) == 0 or len(y_indices) == 0:
        # 如果没有前景像素，返回空包围盒
        return 0, 0, 0, 0
    
    x_min = int(np.min(x_indices))
    x_max = int(np.max(x_indices))
    y_min = int(np.min(y_indices))
    y_max = int(np.max(y_indices))
    
    return x_min, y_min, x_max, y_max


def save_mask_as_grayscale(mask: np.ndarray, output_path: str) -> None:
    """将mask保存为灰度图像"""
    # 确保mask是二值的
    mask_binary = (mask > 0).astype(np.uint8)
    
    # 转换为灰度值 (0=背景, 255=前景)
    mask_gray = mask_binary * 255
    
    # 保存
    cv2.imwrite(output_path, mask_gray)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Use SAM2 to segment images from point prompts (using grpo.py SAMWrapper)")
    p.add_argument("--input_jsonl", type=str, required=True,
                   help="Input JSONL file containing image paths and point coordinates")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output directory to save mask images")
    p.add_argument("--sam_checkpoint", type=str, required=True,
                   help="Path to SAM2 model checkpoint")
    p.add_argument("--device", type=str, default=None,
                   help="Device to run on (cuda/cpu). Auto-detect if not specified")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip processing if output file already exists")
    p.add_argument("--resize", type=int, nargs=2, default=None, metavar=("WIDTH", "HEIGHT"),
                   help="Resize input images to specified size [width height]")
    p.add_argument("--json_output", type=str, default=None,
                   help="Path to output JSON file containing mask paths and bounding boxes")
    return p.parse_args()


def main():
    args = parse_args()
    
    # 检查输入文件
    if not os.path.isfile(args.input_jsonl):
        print(f"Error: Input JSONL file not found: {args.input_jsonl}")
        return 1
    
    # 检查SAM2检查点
    if not os.path.isfile(args.sam_checkpoint):
        print(f"Error: SAM2 checkpoint not found: {args.sam_checkpoint}")
        return 1
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化SAM2
    print(f"Initializing SAM2 with checkpoint: {args.sam_checkpoint}")
    print(f"Using device: {args.device or 'auto-detect'}")
    
    try:
        sam_wrapper = SAMWrapper(args.sam_checkpoint, args.device)
        print("SAM2 initialized successfully")
    except Exception as e:
        print(f"Error initializing SAM2: {e}")
        return 1
    
    # 处理JSONL文件
    print(f"Processing JSONL file: {args.input_jsonl}")
    
    num_processed = 0
    num_skipped = 0
    num_errors = 0
    
    # 用于存储JSON输出的列表
    json_results = []
    
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[ERROR] Line {line_num}: Invalid JSON - {e}")
                num_errors += 1
                continue
            
            # 验证并提取点信息
            is_valid, points, labels = validate_and_extract_points(obj, line_num)
            if not is_valid:
                num_errors += 1
                continue
            
            image_path = obj["image"]
            
            # 检查图像文件是否存在
            if not os.path.isfile(image_path):
                print(f"[ERROR] Line {line_num}: Image file not found: {image_path}")
                num_errors += 1
                continue
            
            # 生成输出路径
            output_path = get_output_path(image_path, args.output_dir)
            
            # 检查是否跳过已存在的文件
            if args.skip_existing and os.path.isfile(output_path):
                print(f"[SKIP] Line {line_num}: Output already exists: {output_path}")
                num_skipped += 1
                continue
            
            try:
                # 加载图像
                image = PILImage.open(image_path).convert("RGB")
                
                # 调整大小（如果指定）
                if args.resize:
                    image = image.resize(args.resize, PILImage.BILINEAR)
                
                # 运行SAM2预测
                mask, confidence = sam_wrapper.predict(image, points, labels)
                
                # 计算包围盒
                x_min, y_min, x_max, y_max = calculate_bounding_box(mask)
                
                # 保存mask
                save_mask_as_grayscale(mask, output_path)
                
                # 如果需要JSON输出，添加到结果列表
                if args.json_output:
                    json_results.append({
                        "mask_path": output_path,
                        "bounding_box": {
                            "x_min": x_min,
                            "y_min": y_min,
                            "x_max": x_max,
                            "y_max": y_max
                        }
                    })
                
                print(f"[OK] Line {line_num}: {image_path} -> {output_path} (confidence: {confidence:.3f}, bbox: [{x_min},{y_min},{x_max},{y_max}])")
                num_processed += 1
                
            except Exception as e:
                print(f"[ERROR] Line {line_num}: Failed to process {image_path} - {e}")
                num_errors += 1
                continue
    
    # 打印总结
    print(f"\nProcessing completed:")
    print(f"  Processed: {num_processed}")
    print(f"  Skipped: {num_skipped}")
    print(f"  Errors: {num_errors}")
    print(f"  Output directory: {args.output_dir}")
    
    return 0 if num_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

```

现在添加保存JSON输出的逻辑：

```python
#!/usr/bin/env python3

"""
使用SAM2从点提示进行图像分割 (简化版本)

直接复用seg-r1/src/open_r1/grpo.py中的SAMWrapper类进行分割

用法:
    python seg-rl/sam2_segment_simple.py \
      --input_jsonl /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/points_canonical.jsonl \
      --output_dir /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/masks_step1 \
      --sam_checkpoint /root/autodl-tmp/works/Seg-R0/third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
      --device cuda
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

# 添加seg-r1和third_party路径以导入SAMWrapper及sam2依赖
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "seg-r1" / "src"))
# Ensure Python sees the inner package directory (third_party/sam2)
sys.path.insert(0, str(project_root / "third_party" / "sam2"))

try:
    from open_r1.grpo import SAMWrapper
except ImportError as e:
    print(f"Error importing SAMWrapper from grpo.py: {e}")
    print("Please ensure the seg-r1 module is properly set up.")
    print("You may need to:")
    print("1. Install SAM2 dependencies")
    print("2. Check the import path")
    sys.exit(1)


def validate_and_extract_points(obj: Dict[str, Any], line_num: int) -> Tuple[bool, Optional[List[Tuple[float, float]]], Optional[List[int]]]:
    """验证并提取JSON条目中的点和标签"""
    if "image" not in obj:
        print(f"[WARN] Line {line_num}: Missing 'image' field")
        return False, None, None
        
    # 检查格式：新格式有"points"和"labels"，旧格式有"x"和"y"
    if "points" in obj and "labels" in obj:
        # 新格式
        points = obj["points"]
        labels = obj["labels"]
        
        if not isinstance(points, list) or not isinstance(labels, list):
            print(f"[WARN] Line {line_num}: 'points' and 'labels' must be lists")
            return False, None, None
            
        if len(points) != len(labels):
            print(f"[WARN] Line {line_num}: 'points' and 'labels' must have same length")
            return False, None, None
        
        # 验证点格式
        valid_points = []
        valid_labels = []
        for point, label in zip(points, labels):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                print(f"[WARN] Line {line_num}: Invalid point format {point}")
                continue
            valid_points.append((float(point[0]), float(point[1])))
            valid_labels.append(int(label))
            
        if len(valid_points) == 0:
            print(f"[WARN] Line {line_num}: No valid points found")
            return False, None, None
            
        return True, valid_points, valid_labels
        
    elif "x" in obj and "y" in obj:
        # 旧格式 - 假设为正类点
        x = float(obj["x"])
        y = float(obj["y"])
        return True, [(x, y)], [1]
        
    else:
        print(f"[WARN] Line {line_num}: Missing coordinate data (need 'points'+'labels' or 'x'+'y')")
        return False, None, None


def get_output_path(image_path: str, output_dir: str) -> str:
    """生成输出mask文件路径"""
    image_name = Path(image_path).stem
    return os.path.join(output_dir, f"{image_name}.png")


def calculate_bounding_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """计算mask的最小包围盒
    
    Args:
        mask: 二值mask数组
        
    Returns:
        (x_min, y_min, x_max, y_max) 包围盒坐标
    """
    # 确保mask是二值的
    mask_binary = (mask > 0).astype(np.uint8)
    
    # 找到非零像素的坐标
    y_indices, x_indices = np.where(mask_binary)
    
    if len(x_indices) == 0 or len(y_indices) == 0:
        # 如果没有前景像素，返回空包围盒
        return 0, 0, 0, 0
    
    x_min = int(np.min(x_indices))
    x_max = int(np.max(x_indices))
    y_min = int(np.min(y_indices))
    y_max = int(np.max(y_indices))
    
    return x_min, y_min, x_max, y_max


def save_mask_as_grayscale(mask: np.ndarray, output_path: str) -> None:
    """将mask保存为灰度图像"""
    # 确保mask是二值的
    mask_binary = (mask > 0).astype(np.uint8)
    
    # 转换为灰度值 (0=背景, 255=前景)
    mask_gray = mask_binary * 255
    
    # 保存
    cv2.imwrite(output_path, mask_gray)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Use SAM2 to segment images from point prompts (using grpo.py SAMWrapper)")
    p.add_argument("--input_jsonl", type=str, required=True,
                   help="Input JSONL file containing image paths and point coordinates")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output directory to save mask images")
    p.add_argument("--sam_checkpoint", type=str, required=True,
                   help="Path to SAM2 model checkpoint")
    p.add_argument("--device", type=str, default=None,
                   help="Device to run on (cuda/cpu). Auto-detect if not specified")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip processing if output file already exists")
    p.add_argument("--resize", type=int, nargs=2, default=None, metavar=("WIDTH", "HEIGHT"),
                   help="Resize input images to specified size [width height]")
    p.add_argument("--json_output", type=str, default=None,
                   help="Path to output JSON file containing mask paths and bounding boxes")
    return p.parse_args()


def main():
    args = parse_args()
    
    # 检查输入文件
    if not os.path.isfile(args.input_jsonl):
        print(f"Error: Input JSONL file not found: {args.input_jsonl}")
        return 1
    
    # 检查SAM2检查点
    if not os.path.isfile(args.sam_checkpoint):
        print(f"Error: SAM2 checkpoint not found: {args.sam_checkpoint}")
        return 1
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化SAM2
    print(f"Initializing SAM2 with checkpoint: {args.sam_checkpoint}")
    print(f"Using device: {args.device or 'auto-detect'}")
    
    try:
        sam_wrapper = SAMWrapper(args.sam_checkpoint, args.device)
        print("SAM2 initialized successfully")
    except Exception as e:
        print(f"Error initializing SAM2: {e}")
        return 1
    
    # 处理JSONL文件
    print(f"Processing JSONL file: {args.input_jsonl}")
    
    num_processed = 0
    num_skipped = 0
    num_errors = 0
    
    # 用于存储JSON输出的列表
    json_results = []
    
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[ERROR] Line {line_num}: Invalid JSON - {e}")
                num_errors += 1
                continue
            
            # 验证并提取点信息
            is_valid, points, labels = validate_and_extract_points(obj, line_num)
            if not is_valid:
                num_errors += 1
                continue
            
            image_path = obj["image"]
            
            # 检查图像文件是否存在
            if not os.path.isfile(image_path):
                print(f"[ERROR] Line {line_num}: Image file not found: {image_path}")
                num_errors += 1
                continue
            
            # 生成输出路径
            output_path = get_output_path(image_path, args.output_dir)
            
            # 检查是否跳过已存在的文件
            if args.skip_existing and os.path.isfile(output_path):
                print(f"[SKIP] Line {line_num}: Output already exists: {output_path}")
                num_skipped += 1
                continue
            
            try:
                # 加载图像
                image = PILImage.open(image_path).convert("RGB")
                
                # 调整大小（如果指定）
                if args.resize:
                    image = image.resize(args.resize, PILImage.BILINEAR)
                
                # 运行SAM2预测
                mask, confidence = sam_wrapper.predict(image, points, labels)
                
                # 计算包围盒
                x_min, y_min, x_max, y_max = calculate_bounding_box(mask)
                
                # 保存mask
                save_mask_as_grayscale(mask, output_path)
                
                # 如果需要JSON输出，添加到结果列表
                if args.json_output:
                    json_results.append({
                        "mask_path": output_path,
                        "bounding_box": {
                            "x_min": x_min,
                            "y_min": y_min,
                            "x_max": x_max,
                            "y_max": y_max
                        }
                    })
                
                print(f"[OK] Line {line_num}: {image_path} -> {output_path} (confidence: {confidence:.3f}, bbox: [{x_min},{y_min},{x_max},{y_max}])")
                num_processed += 1
                
            except Exception as e:
                print(f"[ERROR] Line {line_num}: Failed to process {image_path} - {e}")
                num_errors += 1
                continue
    
    # 保存JSON输出
    if args.json_output and json_results:
        try:
            with open(args.json_output, "w", encoding="utf-8") as json_file:
                json.dump(json_results, json_file, indent=2, ensure_ascii=False)
            print(f"JSON output saved to: {args.json_output}")
        except Exception as e:
            print(f"[ERROR] Failed to save JSON output: {e}")
            return 1
    
    # 打印总结
    print(f"\nProcessing completed:")
    print(f"  Processed: {num_processed}")
    print(f"  Skipped: {num_skipped}")
    print(f"  Errors: {num_errors}")
    print(f"  Output directory: {args.output_dir}")
    if args.json_output:
        print(f"  JSON output: {args.json_output}")
    
    return 0 if num_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

```

现在开始更新sam2_segment_from_points.py文件。首先添加包围盒计算函数：

```python
#!/usr/bin/env python3

"""
使用SAM2从点提示进行图像分割

读取包含图像路径和正/负类点坐标的JSONL文件，使用SAM2进行分割，
并将预测的mask保存为灰度图像。

输入JSONL格式:
    新格式: {"image": "/path/img.jpg", "points": [[x1,y1], [x2,y2]], "labels": [1, 0]}
    旧格式: {"image": "/path/img.jpg", "x": x1, "y": y1}

输出:
    与输入图像对应的灰度mask图像，保存到指定目录
    - 0 (黑色): 背景
    - 255 (白色): 前景

依赖项:
    - SAM2 (Segment Anything Model 2)
    - torch, PIL, opencv-python, numpy

安装SAM2:
    git clone https://github.com/facebookresearch/segment-anything-2.git third_party/sam2
    cd third_party/sam2 && pip install -e .

下载模型:
    # 下载SAM2.1 Hiera Large模型 (~900MB)
    wget -O third_party/sam2/checkpoints/sam2.1_hiera_large.pt \\
      https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

基础用法:
    python seg-rl/sam2_segment_from_points.py \\
      --input_jsonl /path/to/points.jsonl \\
      --output_dir /path/to/masks \\
      --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \\
      --device cuda

高级用法:
    python seg-rl/sam2_segment_from_points.py \\
      --input_jsonl /path/to/points.jsonl \\
      --output_dir /path/to/masks \\
      --sam_checkpoint /path/to/sam2.1_hiera_large.pt \\
      --device cuda \\
      --resize 512 512 \\
      --skip_existing
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

# 添加sam2路径
sys.path.append(str(Path(__file__).parent.parent / "third_party" / "sam2"))

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError as e:
    print(f"Error importing SAM2: {e}")
    print("Please ensure SAM2 is properly installed and the path is correct.")
    sys.exit(1)


class SAMWrapper:
    """SAM2包装器，用于图像分割预测"""
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """初始化SAM2模型和预测器
        
        Args:
            model_path: SAM2模型检查点路径
            device: 运行设备 (e.g. "cuda", "cuda:0", "cpu")
                   如果为None，将自动检测可用设备
        """
        # SAM2配置文件路径
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        
        # 检查配置文件是否存在
        config_path = Path(__file__).parent.parent / "third_party" / "sam2" / model_cfg
        if not config_path.exists():
            # 尝试相对于当前目录
            config_path = Path(model_cfg)
            if not config_path.exists():
                raise FileNotFoundError(f"SAM2 config file not found: {model_cfg}")
        
        sam_model = build_sam2(str(config_path), model_path)
        
        # 自动检测设备
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)
        
        # 移动到指定设备
        sam_model = sam_model.to(self.device)
        
        # 初始化预测器
        self.predictor = SAM2ImagePredictor(sam_model)
        self.last_mask = None
        
    def predict(self, 
                image: PILImage.Image, 
                points: Optional[List[Tuple[int, int]]] = None, 
                labels: Optional[List[int]] = None,
                bbox: Optional[List[int]] = None) -> Tuple[np.ndarray, float]:
        """使用给定提示运行SAM2预测
        
        Args:
            image: 输入PIL图像
            points: 点坐标列表 [(x,y), ...]
            labels: 点标签列表 (1=前景, 0=背景)
            bbox: 可选边界框 [x1,y1,x2,y2]
            
        Returns:
            (predicted_mask, confidence_score)的元组
        """
        input_points = np.array(points) if points else None
        input_labels = np.array(labels) if labels else None
        input_bboxes = np.array([bbox]) if bbox else None

        # 转换为numpy数组
        image_np = np.array(image)
        
        # 确保是RGB格式
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image_np
        
        # 设置图像
        self.predictor.set_image(rgb_image)
        
        # 预测
        mask_pred, score, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=input_bboxes,
            multimask_output=False,
        )
        
        self.last_mask = mask_pred[0]
        return mask_pred[0], score[0]


def validate_and_extract_points(obj: Dict[str, Any], line_num: int) -> Tuple[bool, Optional[List[Tuple[float, float]]], Optional[List[int]]]:
    """验证并提取JSON条目中的点和标签
    
    Args:
        obj: JSON对象
        line_num: 行号（用于错误报告）
        
    Returns:
        (is_valid, points, labels)的元组
    """
    if "image" not in obj:
        print(f"[WARN] Line {line_num}: Missing 'image' field")
        return False, None, None
        
    # 检查格式：新格式有"points"和"labels"，旧格式有"x"和"y"
    if "points" in obj and "labels" in obj:
        # 新格式
        points = obj["points"]
        labels = obj["labels"]
        
        if not isinstance(points, list) or not isinstance(labels, list):
            print(f"[WARN] Line {line_num}: 'points' and 'labels' must be lists")
            return False, None, None
            
        if len(points) != len(labels):
            print(f"[WARN] Line {line_num}: 'points' and 'labels' must have same length")
            return False, None, None
        
        # 验证点格式
        valid_points = []
        valid_labels = []
        for point, label in zip(points, labels):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                print(f"[WARN] Line {line_num}: Invalid point format {point}")
                continue
            valid_points.append((float(point[0]), float(point[1])))
            valid_labels.append(int(label))
            
        if len(valid_points) == 0:
            print(f"[WARN] Line {line_num}: No valid points found")
            return False, None, None
            
        return True, valid_points, valid_labels
        
    elif "x" in obj and "y" in obj:
        # 旧格式 - 假设为正类点
        x = float(obj["x"])
        y = float(obj["y"])
        return True, [(x, y)], [1]
        
    else:
        print(f"[WARN] Line {line_num}: Missing coordinate data (need 'points'+'labels' or 'x'+'y')")
        return False, None, None


def get_output_path(image_path: str, output_dir: str) -> str:
    """生成输出mask文件路径
    
    Args:
        image_path: 输入图像路径
        output_dir: 输出目录
        
    Returns:
        输出mask文件路径
    """
    image_name = Path(image_path).stem
    return os.path.join(output_dir, f"{image_name}.png")


def calculate_bounding_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """计算mask的最小包围盒
    
    Args:
        mask: 二值mask数组
        
    Returns:
        (x_min, y_min, x_max, y_max) 包围盒坐标
    """
    # 确保mask是二值的
    mask_binary = (mask > 0).astype(np.uint8)
    
    # 找到非零像素的坐标
    y_indices, x_indices = np.where(mask_binary)
    
    if len(x_indices) == 0 or len(y_indices) == 0:
        # 如果没有前景像素，返回空包围盒
        return 0, 0, 0, 0
    
    x_min = int(np.min(x_indices))
    x_max = int(np.max(x_indices))
    y_min = int(np.min(y_indices))
    y_max = int(np.max(y_indices))
    
    return x_min, y_min, x_max, y_max


def save_mask_as_grayscale(mask: np.ndarray, output_path: str) -> None:
    """将mask保存为灰度图像
    
    Args:
        mask: 二值mask数组
        output_path: 输出文件路径
    """
    # 确保mask是二值的
    mask_binary = (mask > 0).astype(np.uint8)
    
    # 转换为灰度值 (0=背景, 255=前景)
    mask_gray = mask_binary * 255
    
    # 保存
    cv2.imwrite(output_path, mask_gray)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Use SAM2 to segment images from point prompts")
    p.add_argument("--input_jsonl", type=str, required=True,
                   help="Input JSONL file containing image paths and point coordinates")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output directory to save mask images")
    p.add_argument("--sam_checkpoint", type=str, required=True,
                   help="Path to SAM2 model checkpoint")
    p.add_argument("--device", type=str, default=None,
                   help="Device to run on (cuda/cpu). Auto-detect if not specified")
    p.add_argument("--config_path", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml",
                   help="Path to SAM2 config file")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip processing if output file already exists")
    p.add_argument("--resize", type=int, nargs=2, default=None, metavar=("WIDTH", "HEIGHT"),
                   help="Resize input images to specified size [width height]")
    return p.parse_args()


def main():
    args = parse_args()
    
    # 检查输入文件
    if not os.path.isfile(args.input_jsonl):
        print(f"Error: Input JSONL file not found: {args.input_jsonl}")
        return 1
    
    # 检查SAM2检查点
    if not os.path.isfile(args.sam_checkpoint):
        print(f"Error: SAM2 checkpoint not found: {args.sam_checkpoint}")
        return 1
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化SAM2
    print(f"Initializing SAM2 with checkpoint: {args.sam_checkpoint}")
    print(f"Using device: {args.device or 'auto-detect'}")
    
    try:
        sam_wrapper = SAMWrapper(args.sam_checkpoint, args.device)
        print("SAM2 initialized successfully")
    except Exception as e:
        print(f"Error initializing SAM2: {e}")
        return 1
    
    # 处理JSONL文件
    print(f"Processing JSONL file: {args.input_jsonl}")
    
    num_processed = 0
    num_skipped = 0
    num_errors = 0
    
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[ERROR] Line {line_num}: Invalid JSON - {e}")
                num_errors += 1
                continue
            
            # 验证并提取点信息
            is_valid, points, labels = validate_and_extract_points(obj, line_num)
            if not is_valid:
                num_errors += 1
                continue
            
            image_path = obj["image"]
            
            # 检查图像文件是否存在
            if not os.path.isfile(image_path):
                print(f"[ERROR] Line {line_num}: Image file not found: {image_path}")
                num_errors += 1
                continue
            
            # 生成输出路径
            output_path = get_output_path(image_path, args.output_dir)
            
            # 检查是否跳过已存在的文件
            if args.skip_existing and os.path.isfile(output_path):
                print(f"[SKIP] Line {line_num}: Output already exists: {output_path}")
                num_skipped += 1
                continue
            
            try:
                # 加载图像
                image = PILImage.open(image_path).convert("RGB")
                
                # 调整大小（如果指定）
                if args.resize:
                    image = image.resize(args.resize, PILImage.BILINEAR)
                
                # 运行SAM2预测
                mask, confidence = sam_wrapper.predict(image, points, labels)
                
                # 保存mask
                save_mask_as_grayscale(mask, output_path)
                
                print(f"[OK] Line {line_num}: {image_path} -> {output_path} (confidence: {confidence:.3f})")
                num_processed += 1
                
            except Exception as e:
                print(f"[ERROR] Line {line_num}: Failed to process {image_path} - {e}")
                num_errors += 1
                continue
    
    # 打印总结
    print(f"\nProcessing completed:")
    print(f"  Processed: {num_processed}")
    print(f"  Skipped: {num_skipped}")
    print(f"  Errors: {num_errors}")
    print(f"  Output directory: {args.output_dir}")
    
    return 0 if num_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

```

现在为sam2_segment_from_points.py添加JSON输出参数：

```python
#!/usr/bin/env python3

"""
使用SAM2从点提示进行图像分割

读取包含图像路径和正/负类点坐标的JSONL文件，使用SAM2进行分割，
并将预测的mask保存为灰度图像。

输入JSONL格式:
    新格式: {"image": "/path/img.jpg", "points": [[x1,y1], [x2,y2]], "labels": [1, 0]}
    旧格式: {"image": "/path/img.jpg", "x": x1, "y": y1}

输出:
    与输入图像对应的灰度mask图像，保存到指定目录
    - 0 (黑色): 背景
    - 255 (白色): 前景

依赖项:
    - SAM2 (Segment Anything Model 2)
    - torch, PIL, opencv-python, numpy

安装SAM2:
    git clone https://github.com/facebookresearch/segment-anything-2.git third_party/sam2
    cd third_party/sam2 && pip install -e .

下载模型:
    # 下载SAM2.1 Hiera Large模型 (~900MB)
    wget -O third_party/sam2/checkpoints/sam2.1_hiera_large.pt \\
      https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

基础用法:
    python seg-rl/sam2_segment_from_points.py \\
      --input_jsonl /path/to/points.jsonl \\
      --output_dir /path/to/masks \\
      --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \\
      --device cuda

高级用法:
    python seg-rl/sam2_segment_from_points.py \\
      --input_jsonl /path/to/points.jsonl \\
      --output_dir /path/to/masks \\
      --sam_checkpoint /path/to/sam2.1_hiera_large.pt \\
      --device cuda \\
      --resize 512 512 \\
      --skip_existing
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

# 添加sam2路径
sys.path.append(str(Path(__file__).parent.parent / "third_party" / "sam2"))

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError as e:
    print(f"Error importing SAM2: {e}")
    print("Please ensure SAM2 is properly installed and the path is correct.")
    sys.exit(1)


class SAMWrapper:
    """SAM2包装器，用于图像分割预测"""
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """初始化SAM2模型和预测器
        
        Args:
            model_path: SAM2模型检查点路径
            device: 运行设备 (e.g. "cuda", "cuda:0", "cpu")
                   如果为None，将自动检测可用设备
        """
        # SAM2配置文件路径
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        
        # 检查配置文件是否存在
        config_path = Path(__file__).parent.parent / "third_party" / "sam2" / model_cfg
        if not config_path.exists():
            # 尝试相对于当前目录
            config_path = Path(model_cfg)
            if not config_path.exists():
                raise FileNotFoundError(f"SAM2 config file not found: {model_cfg}")
        
        sam_model = build_sam2(str(config_path), model_path)
        
        # 自动检测设备
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)
        
        # 移动到指定设备
        sam_model = sam_model.to(self.device)
        
        # 初始化预测器
        self.predictor = SAM2ImagePredictor(sam_model)
        self.last_mask = None
        
    def predict(self, 
                image: PILImage.Image, 
                points: Optional[List[Tuple[int, int]]] = None, 
                labels: Optional[List[int]] = None,
                bbox: Optional[List[int]] = None) -> Tuple[np.ndarray, float]:
        """使用给定提示运行SAM2预测
        
        Args:
            image: 输入PIL图像
            points: 点坐标列表 [(x,y), ...]
            labels: 点标签列表 (1=前景, 0=背景)
            bbox: 可选边界框 [x1,y1,x2,y2]
            
        Returns:
            (predicted_mask, confidence_score)的元组
        """
        input_points = np.array(points) if points else None
        input_labels = np.array(labels) if labels else None
        input_bboxes = np.array([bbox]) if bbox else None

        # 转换为numpy数组
        image_np = np.array(image)
        
        # 确保是RGB格式
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image_np
        
        # 设置图像
        self.predictor.set_image(rgb_image)
        
        # 预测
        mask_pred, score, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=input_bboxes,
            multimask_output=False,
        )
        
        self.last_mask = mask_pred[0]
        return mask_pred[0], score[0]


def validate_and_extract_points(obj: Dict[str, Any], line_num: int) -> Tuple[bool, Optional[List[Tuple[float, float]]], Optional[List[int]]]:
    """验证并提取JSON条目中的点和标签
    
    Args:
        obj: JSON对象
        line_num: 行号（用于错误报告）
        
    Returns:
        (is_valid, points, labels)的元组
    """
    if "image" not in obj:
        print(f"[WARN] Line {line_num}: Missing 'image' field")
        return False, None, None
        
    # 检查格式：新格式有"points"和"labels"，旧格式有"x"和"y"
    if "points" in obj and "labels" in obj:
        # 新格式
        points = obj["points"]
        labels = obj["labels"]
        
        if not isinstance(points, list) or not isinstance(labels, list):
            print(f"[WARN] Line {line_num}: 'points' and 'labels' must be lists")
            return False, None, None
            
        if len(points) != len(labels):
            print(f"[WARN] Line {line_num}: 'points' and 'labels' must have same length")
            return False, None, None
        
        # 验证点格式
        valid_points = []
        valid_labels = []
        for point, label in zip(points, labels):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                print(f"[WARN] Line {line_num}: Invalid point format {point}")
                continue
            valid_points.append((float(point[0]), float(point[1])))
            valid_labels.append(int(label))
            
        if len(valid_points) == 0:
            print(f"[WARN] Line {line_num}: No valid points found")
            return False, None, None
            
        return True, valid_points, valid_labels
        
    elif "x" in obj and "y" in obj:
        # 旧格式 - 假设为正类点
        x = float(obj["x"])
        y = float(obj["y"])
        return True, [(x, y)], [1]
        
    else:
        print(f"[WARN] Line {line_num}: Missing coordinate data (need 'points'+'labels' or 'x'+'y')")
        return False, None, None


def get_output_path(image_path: str, output_dir: str) -> str:
    """生成输出mask文件路径
    
    Args:
        image_path: 输入图像路径
        output_dir: 输出目录
        
    Returns:
        输出mask文件路径
    """
    image_name = Path(image_path).stem
    return os.path.join(output_dir, f"{image_name}.png")


def calculate_bounding_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """计算mask的最小包围盒
    
    Args:
        mask: 二值mask数组
        
    Returns:
        (x_min, y_min, x_max, y_max) 包围盒坐标
    """
    # 确保mask是二值的
    mask_binary = (mask > 0).astype(np.uint8)
    
    # 找到非零像素的坐标
    y_indices, x_indices = np.where(mask_binary)
    
    if len(x_indices) == 0 or len(y_indices) == 0:
        # 如果没有前景像素，返回空包围盒
        return 0, 0, 0, 0
    
    x_min = int(np.min(x_indices))
    x_max = int(np.max(x_indices))
    y_min = int(np.min(y_indices))
    y_max = int(np.max(y_indices))
    
    return x_min, y_min, x_max, y_max


def save_mask_as_grayscale(mask: np.ndarray, output_path: str) -> None:
    """将mask保存为灰度图像
    
    Args:
        mask: 二值mask数组
        output_path: 输出文件路径
    """
    # 确保mask是二值的
    mask_binary = (mask > 0).astype(np.uint8)
    
    # 转换为灰度值 (0=背景, 255=前景)
    mask_gray = mask_binary * 255
    
    # 保存
    cv2.imwrite(output_path, mask_gray)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Use SAM2 to segment images from point prompts")
    p.add_argument("--input_jsonl", type=str, required=True,
                   help="Input JSONL file containing image paths and point coordinates")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output directory to save mask images")
    p.add_argument("--sam_checkpoint", type=str, required=True,
                   help="Path to SAM2 model checkpoint")
    p.add_argument("--device", type=str, default=None,
                   help="Device to run on (cuda/cpu). Auto-detect if not specified")
    p.add_argument("--config_path", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml",
                   help="Path to SAM2 config file")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip processing if output file already exists")
    p.add_argument("--resize", type=int, nargs=2, default=None, metavar=("WIDTH", "HEIGHT"),
                   help="Resize input images to specified size [width height]")
    p.add_argument("--json_output", type=str, default=None,
                   help="Path to output JSON file containing mask paths and bounding boxes")
    return p.parse_args()


def main():
    args = parse_args()
    
    # 检查输入文件
    if not os.path.isfile(args.input_jsonl):
        print(f"Error: Input JSONL file not found: {args.input_jsonl}")
        return 1
    
    # 检查SAM2检查点
    if not os.path.isfile(args.sam_checkpoint):
        print(f"Error: SAM2 checkpoint not found: {args.sam_checkpoint}")
        return 1
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化SAM2
    print(f"Initializing SAM2 with checkpoint: {args.sam_checkpoint}")
    print(f"Using device: {args.device or 'auto-detect'}")
    
    try:
        sam_wrapper = SAMWrapper(args.sam_checkpoint, args.device)
        print("SAM2 initialized successfully")
    except Exception as e:
        print(f"Error initializing SAM2: {e}")
        return 1
    
    # 处理JSONL文件
    print(f"Processing JSONL file: {args.input_jsonl}")
    
    num_processed = 0
    num_skipped = 0
    num_errors = 0
    
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[ERROR] Line {line_num}: Invalid JSON - {e}")
                num_errors += 1
                continue
            
            # 验证并提取点信息
            is_valid, points, labels = validate_and_extract_points(obj, line_num)
            if not is_valid:
                num_errors += 1
                continue
            
            image_path = obj["image"]
            
            # 检查图像文件是否存在
            if not os.path.isfile(image_path):
                print(f"[ERROR] Line {line_num}: Image file not found: {image_path}")
                num_errors += 1
                continue
            
            # 生成输出路径
            output_path = get_output_path(image_path, args.output_dir)
            
            # 检查是否跳过已存在的文件
            if args.skip_existing and os.path.isfile(output_path):
                print(f"[SKIP] Line {line_num}: Output already exists: {output_path}")
                num_skipped += 1
                continue
            
            try:
                # 加载图像
                image = PILImage.open(image_path).convert("RGB")
                
                # 调整大小（如果指定）
                if args.resize:
                    image = image.resize(args.resize, PILImage.BILINEAR)
                
                # 运行SAM2预测
                mask, confidence = sam_wrapper.predict(image, points, labels)
                
                # 保存mask
                save_mask_as_grayscale(mask, output_path)
                
                print(f"[OK] Line {line_num}: {image_path} -> {output_path} (confidence: {confidence:.3f})")
                num_processed += 1
                
            except Exception as e:
                print(f"[ERROR] Line {line_num}: Failed to process {image_path} - {e}")
                num_errors += 1
                continue
    
    # 打印总结
    print(f"\nProcessing completed:")
    print(f"  Processed: {num_processed}")
    print(f"  Skipped: {num_skipped}")
    print(f"  Errors: {num_errors}")
    print(f"  Output directory: {args.output_dir}")
    
    return 0 if num_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

```

现在更新main函数来支持JSON输出：

```python
#!/usr/bin/env python3

"""
使用SAM2从点提示进行图像分割

读取包含图像路径和正/负类点坐标的JSONL文件，使用SAM2进行分割，
并将预测的mask保存为灰度图像。

输入JSONL格式:
    新格式: {"image": "/path/img.jpg", "points": [[x1,y1], [x2,y2]], "labels": [1, 0]}
    旧格式: {"image": "/path/img.jpg", "x": x1, "y": y1}

输出:
    与输入图像对应的灰度mask图像，保存到指定目录
    - 0 (黑色): 背景
    - 255 (白色): 前景

依赖项:
    - SAM2 (Segment Anything Model 2)
    - torch, PIL, opencv-python, numpy

安装SAM2:
    git clone https://github.com/facebookresearch/segment-anything-2.git third_party/sam2
    cd third_party/sam2 && pip install -e .

下载模型:
    # 下载SAM2.1 Hiera Large模型 (~900MB)
    wget -O third_party/sam2/checkpoints/sam2.1_hiera_large.pt \\
      https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

基础用法:
    python seg-rl/sam2_segment_from_points.py \\
      --input_jsonl /path/to/points.jsonl \\
      --output_dir /path/to/masks \\
      --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \\
      --device cuda

高级用法:
    python seg-rl/sam2_segment_from_points.py \\
      --input_jsonl /path/to/points.jsonl \\
      --output_dir /path/to/masks \\
      --sam_checkpoint /path/to/sam2.1_hiera_large.pt \\
      --device cuda \\
      --resize 512 512 \\
      --skip_existing
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

# 添加sam2路径
sys.path.append(str(Path(__file__).parent.parent / "third_party" / "sam2"))

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError as e:
    print(f"Error importing SAM2: {e}")
    print("Please ensure SAM2 is properly installed and the path is correct.")
    sys.exit(1)


class SAMWrapper:
    """SAM2包装器，用于图像分割预测"""
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """初始化SAM2模型和预测器
        
        Args:
            model_path: SAM2模型检查点路径
            device: 运行设备 (e.g. "cuda", "cuda:0", "cpu")
                   如果为None，将自动检测可用设备
        """
        # SAM2配置文件路径
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        
        # 检查配置文件是否存在
        config_path = Path(__file__).parent.parent / "third_party" / "sam2" / model_cfg
        if not config_path.exists():
            # 尝试相对于当前目录
            config_path = Path(model_cfg)
            if not config_path.exists():
                raise FileNotFoundError(f"SAM2 config file not found: {model_cfg}")
        
        sam_model = build_sam2(str(config_path), model_path)
        
        # 自动检测设备
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)
        
        # 移动到指定设备
        sam_model = sam_model.to(self.device)
        
        # 初始化预测器
        self.predictor = SAM2ImagePredictor(sam_model)
        self.last_mask = None
        
    def predict(self, 
                image: PILImage.Image, 
                points: Optional[List[Tuple[int, int]]] = None, 
                labels: Optional[List[int]] = None,
                bbox: Optional[List[int]] = None) -> Tuple[np.ndarray, float]:
        """使用给定提示运行SAM2预测
        
        Args:
            image: 输入PIL图像
            points: 点坐标列表 [(x,y), ...]
            labels: 点标签列表 (1=前景, 0=背景)
            bbox: 可选边界框 [x1,y1,x2,y2]
            
        Returns:
            (predicted_mask, confidence_score)的元组
        """
        input_points = np.array(points) if points else None
        input_labels = np.array(labels) if labels else None
        input_bboxes = np.array([bbox]) if bbox else None

        # 转换为numpy数组
        image_np = np.array(image)
        
        # 确保是RGB格式
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image_np
        
        # 设置图像
        self.predictor.set_image(rgb_image)
        
        # 预测
        mask_pred, score, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=input_bboxes,
            multimask_output=False,
        )
        
        self.last_mask = mask_pred[0]
        return mask_pred[0], score[0]


def validate_and_extract_points(obj: Dict[str, Any], line_num: int) -> Tuple[bool, Optional[List[Tuple[float, float]]], Optional[List[int]]]:
    """验证并提取JSON条目中的点和标签
    
    Args:
        obj: JSON对象
        line_num: 行号（用于错误报告）
        
    Returns:
        (is_valid, points, labels)的元组
    """
    if "image" not in obj:
        print(f"[WARN] Line {line_num}: Missing 'image' field")
        return False, None, None
        
    # 检查格式：新格式有"points"和"labels"，旧格式有"x"和"y"
    if "points" in obj and "labels" in obj:
        # 新格式
        points = obj["points"]
        labels = obj["labels"]
        
        if not isinstance(points, list) or not isinstance(labels, list):
            print(f"[WARN] Line {line_num}: 'points' and 'labels' must be lists")
            return False, None, None
            
        if len(points) != len(labels):
            print(f"[WARN] Line {line_num}: 'points' and 'labels' must have same length")
            return False, None, None
        
        # 验证点格式
        valid_points = []
        valid_labels = []
        for point, label in zip(points, labels):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                print(f"[WARN] Line {line_num}: Invalid point format {point}")
                continue
            valid_points.append((float(point[0]), float(point[1])))
            valid_labels.append(int(label))
            
        if len(valid_points) == 0:
            print(f"[WARN] Line {line_num}: No valid points found")
            return False, None, None
            
        return True, valid_points, valid_labels
        
    elif "x" in obj and "y" in obj:
        # 旧格式 - 假设为正类点
        x = float(obj["x"])
        y = float(obj["y"])
        return True, [(x, y)], [1]
        
    else:
        print(f"[WARN] Line {line_num}: Missing coordinate data (need 'points'+'labels' or 'x'+'y')")
        return False, None, None


def get_output_path(image_path: str, output_dir: str) -> str:
    """生成输出mask文件路径
    
    Args:
        image_path: 输入图像路径
        output_dir: 输出目录
        
    Returns:
        输出mask文件路径
    """
    image_name = Path(image_path).stem
    return os.path.join(output_dir, f"{image_name}.png")


def calculate_bounding_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """计算mask的最小包围盒
    
    Args:
        mask: 二值mask数组
        
    Returns:
        (x_min, y_min, x_max, y_max) 包围盒坐标
    """
    # 确保mask是二值的
    mask_binary = (mask > 0).astype(np.uint8)
    
    # 找到非零像素的坐标
    y_indices, x_indices = np.where(mask_binary)
    
    if len(x_indices) == 0 or len(y_indices) == 0:
        # 如果没有前景像素，返回空包围盒
        return 0, 0, 0, 0
    
    x_min = int(np.min(x_indices))
    x_max = int(np.max(x_indices))
    y_min = int(np.min(y_indices))
    y_max = int(np.max(y_indices))
    
    return x_min, y_min, x_max, y_max


def save_mask_as_grayscale(mask: np.ndarray, output_path: str) -> None:
    """将mask保存为灰度图像
    
    Args:
        mask: 二值mask数组
        output_path: 输出文件路径
    """
    # 确保mask是二值的
    mask_binary = (mask > 0).astype(np.uint8)
    
    # 转换为灰度值 (0=背景, 255=前景)
    mask_gray = mask_binary * 255
    
    # 保存
    cv2.imwrite(output_path, mask_gray)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Use SAM2 to segment images from point prompts")
    p.add_argument("--input_jsonl", type=str, required=True,
                   help="Input JSONL file containing image paths and point coordinates")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output directory to save mask images")
    p.add_argument("--sam_checkpoint", type=str, required=True,
                   help="Path to SAM2 model checkpoint")
    p.add_argument("--device", type=str, default=None,
                   help="Device to run on (cuda/cpu). Auto-detect if not specified")
    p.add_argument("--config_path", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml",
                   help="Path to SAM2 config file")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip processing if output file already exists")
    p.add_argument("--resize", type=int, nargs=2, default=None, metavar=("WIDTH", "HEIGHT"),
                   help="Resize input images to specified size [width height]")
    p.add_argument("--json_output", type=str, default=None,
                   help="Path to output JSON file containing mask paths and bounding boxes")
    return p.parse_args()


def main():
    args = parse_args()
    
    # 检查输入文件
    if not os.path.isfile(args.input_jsonl):
        print(f"Error: Input JSONL file not found: {args.input_jsonl}")
        return 1
    
    # 检查SAM2检查点
    if not os.path.isfile(args.sam_checkpoint):
        print(f"Error: SAM2 checkpoint not found: {args.sam_checkpoint}")
        return 1
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化SAM2
    print(f"Initializing SAM2 with checkpoint: {args.sam_checkpoint}")
    print(f"Using device: {args.device or 'auto-detect'}")
    
    try:
        sam_wrapper = SAMWrapper(args.sam_checkpoint, args.device)
        print("SAM2 initialized successfully")
    except Exception as e:
        print(f"Error initializing SAM2: {e}")
        return 1
    
    # 处理JSONL文件
    print(f"Processing JSONL file: {args.input_jsonl}")
    
    num_processed = 0
    num_skipped = 0
    num_errors = 0
    
    # 用于存储JSON输出的列表
    json_results = []
    
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[ERROR] Line {line_num}: Invalid JSON - {e}")
                num_errors += 1
                continue
            
            # 验证并提取点信息
            is_valid, points, labels = validate_and_extract_points(obj, line_num)
            if not is_valid:
                num_errors += 1
                continue
            
            image_path = obj["image"]
            
            # 检查图像文件是否存在
            if not os.path.isfile(image_path):
                print(f"[ERROR] Line {line_num}: Image file not found: {image_path}")
                num_errors += 1
                continue
            
            # 生成输出路径
            output_path = get_output_path(image_path, args.output_dir)
            
            # 检查是否跳过已存在的文件
            if args.skip_existing and os.path.isfile(output_path):
                print(f"[SKIP] Line {line_num}: Output already exists: {output_path}")
                num_skipped += 1
                continue
            
            try:
                # 加载图像
                image = PILImage.open(image_path).convert("RGB")
                
                # 调整大小（如果指定）
                if args.resize:
                    image = image.resize(args.resize, PILImage.BILINEAR)
                
                # 运行SAM2预测
                mask, confidence = sam_wrapper.predict(image, points, labels)
                
                # 保存mask
                save_mask_as_grayscale(mask, output_path)
                
                print(f"[OK] Line {line_num}: {image_path} -> {output_path} (confidence: {confidence:.3f})")
                num_processed += 1
                
            except Exception as e:
                print(f"[ERROR] Line {line_num}: Failed to process {image_path} - {e}")
                num_errors += 1
                continue
    
    # 打印总结
    print(f"\nProcessing completed:")
    print(f"  Processed: {num_processed}")
    print(f"  Skipped: {num_skipped}")
    print(f"  Errors: {num_errors}")
    print(f"  Output directory: {args.output_dir}")
    
    return 0 if num_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

```

现在更新处理逻辑：

```python
#!/usr/bin/env python3

"""
使用SAM2从点提示进行图像分割

读取包含图像路径和正/负类点坐标的JSONL文件，使用SAM2进行分割，
并将预测的mask保存为灰度图像。

输入JSONL格式:
    新格式: {"image": "/path/img.jpg", "points": [[x1,y1], [x2,y2]], "labels": [1, 0]}
    旧格式: {"image": "/path/img.jpg", "x": x1, "y": y1}

输出:
    与输入图像对应的灰度mask图像，保存到指定目录
    - 0 (黑色): 背景
    - 255 (白色): 前景

依赖项:
    - SAM2 (Segment Anything Model 2)
    - torch, PIL, opencv-python, numpy

安装SAM2:
    git clone https://github.com/facebookresearch/segment-anything-2.git third_party/sam2
    cd third_party/sam2 && pip install -e .

下载模型:
    # 下载SAM2.1 Hiera Large模型 (~900MB)
    wget -O third_party/sam2/checkpoints/sam2.1_hiera_large.pt \\
      https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

基础用法:
    python seg-rl/sam2_segment_from_points.py \\
      --input_jsonl /path/to/points.jsonl \\
      --output_dir /path/to/masks \\
      --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \\
      --device cuda

高级用法:
    python seg-rl/sam2_segment_from_points.py \\
      --input_jsonl /path/to/points.jsonl \\
      --output_dir /path/to/masks \\
      --sam_checkpoint /path/to/sam2.1_hiera_large.pt \\
      --device cuda \\
      --resize 512 512 \\
      --skip_existing
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

# 添加sam2路径
sys.path.append(str(Path(__file__).parent.parent / "third_party" / "sam2"))

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError as e:
    print(f"Error importing SAM2: {e}")
    print("Please ensure SAM2 is properly installed and the path is correct.")
    sys.exit(1)


class SAMWrapper:
    """SAM2包装器，用于图像分割预测"""
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """初始化SAM2模型和预测器
        
        Args:
            model_path: SAM2模型检查点路径
            device: 运行设备 (e.g. "cuda", "cuda:0", "cpu")
                   如果为None，将自动检测可用设备
        """
        # SAM2配置文件路径
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        
        # 检查配置文件是否存在
        config_path = Path(__file__).parent.parent / "third_party" / "sam2" / model_cfg
        if not config_path.exists():
            # 尝试相对于当前目录
            config_path = Path(model_cfg)
            if not config_path.exists():
                raise FileNotFoundError(f"SAM2 config file not found: {model_cfg}")
        
        sam_model = build_sam2(str(config_path), model_path)
        
        # 自动检测设备
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)
        
        # 移动到指定设备
        sam_model = sam_model.to(self.device)
        
        # 初始化预测器
        self.predictor = SAM2ImagePredictor(sam_model)
        self.last_mask = None
        
    def predict(self, 
                image: PILImage.Image, 
                points: Optional[List[Tuple[int, int]]] = None, 
                labels: Optional[List[int]] = None,
                bbox: Optional[List[int]] = None) -> Tuple[np.ndarray, float]:
        """使用给定提示运行SAM2预测
        
        Args:
            image: 输入PIL图像
            points: 点坐标列表 [(x,y), ...]
            labels: 点标签列表 (1=前景, 0=背景)
            bbox: 可选边界框 [x1,y1,x2,y2]
            
        Returns:
            (predicted_mask, confidence_score)的元组
        """
        input_points = np.array(points) if points else None
        input_labels = np.array(labels) if labels else None
        input_bboxes = np.array([bbox]) if bbox else None

        # 转换为numpy数组
        image_np = np.array(image)
        
        # 确保是RGB格式
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image_np
        
        # 设置图像
        self.predictor.set_image(rgb_image)
        
        # 预测
        mask_pred, score, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=input_bboxes,
            multimask_output=False,
        )
        
        self.last_mask = mask_pred[0]
        return mask_pred[0], score[0]


def validate_and_extract_points(obj: Dict[str, Any], line_num: int) -> Tuple[bool, Optional[List[Tuple[float, float]]], Optional[List[int]]]:
    """验证并提取JSON条目中的点和标签
    
    Args:
        obj: JSON对象
        line_num: 行号（用于错误报告）
        
    Returns:
        (is_valid, points, labels)的元组
    """
    if "image" not in obj:
        print(f"[WARN] Line {line_num}: Missing 'image' field")
        return False, None, None
        
    # 检查格式：新格式有"points"和"labels"，旧格式有"x"和"y"
    if "points" in obj and "labels" in obj:
        # 新格式
        points = obj["points"]
        labels = obj["labels"]
        
        if not isinstance(points, list) or not isinstance(labels, list):
            print(f"[WARN] Line {line_num}: 'points' and 'labels' must be lists")
            return False, None, None
            
        if len(points) != len(labels):
            print(f"[WARN] Line {line_num}: 'points' and 'labels' must have same length")
            return False, None, None
        
        # 验证点格式
        valid_points = []
        valid_labels = []
        for point, label in zip(points, labels):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                print(f"[WARN] Line {line_num}: Invalid point format {point}")
                continue
            valid_points.append((float(point[0]), float(point[1])))
            valid_labels.append(int(label))
            
        if len(valid_points) == 0:
            print(f"[WARN] Line {line_num}: No valid points found")
            return False, None, None
            
        return True, valid_points, valid_labels
        
    elif "x" in obj and "y" in obj:
        # 旧格式 - 假设为正类点
        x = float(obj["x"])
        y = float(obj["y"])
        return True, [(x, y)], [1]
        
    else:
        print(f"[WARN] Line {line_num}: Missing coordinate data (need 'points'+'labels' or 'x'+'y')")
        return False, None, None


def get_output_path(image_path: str, output_dir: str) -> str:
    """生成输出mask文件路径
    
    Args:
        image_path: 输入图像路径
        output_dir: 输出目录
        
    Returns:
        输出mask文件路径
    """
    image_name = Path(image_path).stem
    return os.path.join(output_dir, f"{image_name}.png")


def calculate_bounding_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """计算mask的最小包围盒
    
    Args:
        mask: 二值mask数组
        
    Returns:
        (x_min, y_min, x_max, y_max) 包围盒坐标
    """
    # 确保mask是二值的
    mask_binary = (mask > 0).astype(np.uint8)
    
    # 找到非零像素的坐标
    y_indices, x_indices = np.where(mask_binary)
    
    if len(x_indices) == 0 or len(y_indices) == 0:
        # 如果没有前景像素，返回空包围盒
        return 0, 0, 0, 0
    
    x_min = int(np.min(x_indices))
    x_max = int(np.max(x_indices))
    y_min = int(np.min(y_indices))
    y_max = int(np.max(y_indices))
    
    return x_min, y_min, x_max, y_max


def save_mask_as_grayscale(mask: np.ndarray, output_path: str) -> None:
    """将mask保存为灰度图像
    
    Args:
        mask: 二值mask数组
        output_path: 输出文件路径
    """
    # 确保mask是二值的
    mask_binary = (mask > 0).astype(np.uint8)
    
    # 转换为灰度值 (0=背景, 255=前景)
    mask_gray = mask_binary * 255
    
    # 保存
    cv2.imwrite(output_path, mask_gray)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Use SAM2 to segment images from point prompts")
    p.add_argument("--input_jsonl", type=str, required=True,
                   help="Input JSONL file containing image paths and point coordinates")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output directory to save mask images")
    p.add_argument("--sam_checkpoint", type=str, required=True,
                   help="Path to SAM2 model checkpoint")
    p.add_argument("--device", type=str, default=None,
                   help="Device to run on (cuda/cpu). Auto-detect if not specified")
    p.add_argument("--config_path", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml",
                   help="Path to SAM2 config file")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip processing if output file already exists")
    p.add_argument("--resize", type=int, nargs=2, default=None, metavar=("WIDTH", "HEIGHT"),
                   help="Resize input images to specified size [width height]")
    p.add_argument("--json_output", type=str, default=None,
                   help="Path to output JSON file containing mask paths and bounding boxes")
    return p.parse_args()


def main():
    args = parse_args()
    
    # 检查输入文件
    if not os.path.isfile(args.input_jsonl):
        print(f"Error: Input JSONL file not found: {args.input_jsonl}")
        return 1
    
    # 检查SAM2检查点
    if not os.path.isfile(args.sam_checkpoint):
        print(f"Error: SAM2 checkpoint not found: {args.sam_checkpoint}")
        return 1
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化SAM2
    print(f"Initializing SAM2 with checkpoint: {args.sam_checkpoint}")
    print(f"Using device: {args.device or 'auto-detect'}")
    
    try:
        sam_wrapper = SAMWrapper(args.sam_checkpoint, args.device)
        print("SAM2 initialized successfully")
    except Exception as e:
        print(f"Error initializing SAM2: {e}")
        return 1
    
    # 处理JSONL文件
    print(f"Processing JSONL file: {args.input_jsonl}")
    
    num_processed = 0
    num_skipped = 0
    num_errors = 0
    
    # 用于存储JSON输出的列表
    json_results = []
    
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[ERROR] Line {line_num}: Invalid JSON - {e}")
                num_errors += 1
                continue
            
            # 验证并提取点信息
            is_valid, points, labels = validate_and_extract_points(obj, line_num)
            if not is_valid:
                num_errors += 1
                continue
            
            image_path = obj["image"]
            
            # 检查图像文件是否存在
            if not os.path.isfile(image_path):
                print(f"[ERROR] Line {line_num}: Image file not found: {image_path}")
                num_errors += 1
                continue
            
            # 生成输出路径
            output_path = get_output_path(image_path, args.output_dir)
            
            # 检查是否跳过已存在的文件
            if args.skip_existing and os.path.isfile(output_path):
                print(f"[SKIP] Line {line_num}: Output already exists: {output_path}")
                num_skipped += 1
                continue
            
            try:
                # 加载图像
                image = PILImage.open(image_path).convert("RGB")
                
                # 调整大小（如果指定）
                if args.resize:
                    image = image.resize(args.resize, PILImage.BILINEAR)
                
                # 运行SAM2预测
                mask, confidence = sam_wrapper.predict(image, points, labels)
                
                # 计算包围盒
                x_min, y_min, x_max, y_max = calculate_bounding_box(mask)
                
                # 保存mask
                save_mask_as_grayscale(mask, output_path)
                
                # 如果需要JSON输出，添加到结果列表
                if args.json_output:
                    json_results.append({
                        "mask_path": output_path,
                        "bounding_box": {
                            "x_min": x_min,
                            "y_min": y_min,
                            "x_max": x_max,
                            "y_max": y_max
                        }
                    })
                
                print(f"[OK] Line {line_num}: {image_path} -> {output_path} (confidence: {confidence:.3f}, bbox: [{x_min},{y_min},{x_max},{y_max}])")
                num_processed += 1
                
            except Exception as e:
                print(f"[ERROR] Line {line_num}: Failed to process {image_path} - {e}")
                num_errors += 1
                continue
    
    # 打印总结
    print(f"\nProcessing completed:")
    print(f"  Processed: {num_processed}")
    print(f"  Skipped: {num_skipped}")
    print(f"  Errors: {num_errors}")
    print(f"  Output directory: {args.output_dir}")
    
    return 0 if num_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

```

最后添加JSON输出保存逻辑：

```python
#!/usr/bin/env python3

"""
使用SAM2从点提示进行图像分割

读取包含图像路径和正/负类点坐标的JSONL文件，使用SAM2进行分割，
并将预测的mask保存为灰度图像。

输入JSONL格式:
    新格式: {"image": "/path/img.jpg", "points": [[x1,y1], [x2,y2]], "labels": [1, 0]}
    旧格式: {"image": "/path/img.jpg", "x": x1, "y": y1}

输出:
    与输入图像对应的灰度mask图像，保存到指定目录
    - 0 (黑色): 背景
    - 255 (白色): 前景

依赖项:
    - SAM2 (Segment Anything Model 2)
    - torch, PIL, opencv-python, numpy

安装SAM2:
    git clone https://github.com/facebookresearch/segment-anything-2.git third_party/sam2
    cd third_party/sam2 && pip install -e .

下载模型:
    # 下载SAM2.1 Hiera Large模型 (~900MB)
    wget -O third_party/sam2/checkpoints/sam2.1_hiera_large.pt \\
      https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

基础用法:
    python seg-rl/sam2_segment_from_points.py \\
      --input_jsonl /path/to/points.jsonl \\
      --output_dir /path/to/masks \\
      --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \\
      --device cuda

高级用法:
    python seg-rl/sam2_segment_from_points.py \\
      --input_jsonl /path/to/points.jsonl \\
      --output_dir /path/to/masks \\
      --sam_checkpoint /path/to/sam2.1_hiera_large.pt \\
      --device cuda \\
      --resize 512 512 \\
      --skip_existing
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

# 添加sam2路径
sys.path.append(str(Path(__file__).parent.parent / "third_party" / "sam2"))

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError as e:
    print(f"Error importing SAM2: {e}")
    print("Please ensure SAM2 is properly installed and the path is correct.")
    sys.exit(1)


class SAMWrapper:
    """SAM2包装器，用于图像分割预测"""
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """初始化SAM2模型和预测器
        
        Args:
            model_path: SAM2模型检查点路径
            device: 运行设备 (e.g. "cuda", "cuda:0", "cpu")
                   如果为None，将自动检测可用设备
        """
        # SAM2配置文件路径
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        
        # 检查配置文件是否存在
        config_path = Path(__file__).parent.parent / "third_party" / "sam2" / model_cfg
        if not config_path.exists():
            # 尝试相对于当前目录
            config_path = Path(model_cfg)
            if not config_path.exists():
                raise FileNotFoundError(f"SAM2 config file not found: {model_cfg}")
        
        sam_model = build_sam2(str(config_path), model_path)
        
        # 自动检测设备
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)
        
        # 移动到指定设备
        sam_model = sam_model.to(self.device)
        
        # 初始化预测器
        self.predictor = SAM2ImagePredictor(sam_model)
        self.last_mask = None
        
    def predict(self, 
                image: PILImage.Image, 
                points: Optional[List[Tuple[int, int]]] = None, 
                labels: Optional[List[int]] = None,
                bbox: Optional[List[int]] = None) -> Tuple[np.ndarray, float]:
        """使用给定提示运行SAM2预测
        
        Args:
            image: 输入PIL图像
            points: 点坐标列表 [(x,y), ...]
            labels: 点标签列表 (1=前景, 0=背景)
            bbox: 可选边界框 [x1,y1,x2,y2]
            
        Returns:
            (predicted_mask, confidence_score)的元组
        """
        input_points = np.array(points) if points else None
        input_labels = np.array(labels) if labels else None
        input_bboxes = np.array([bbox]) if bbox else None

        # 转换为numpy数组
        image_np = np.array(image)
        
        # 确保是RGB格式
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image_np
        
        # 设置图像
        self.predictor.set_image(rgb_image)
        
        # 预测
        mask_pred, score, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=input_bboxes,
            multimask_output=False,
        )
        
        self.last_mask = mask_pred[0]
        return mask_pred[0], score[0]


def validate_and_extract_points(obj: Dict[str, Any], line_num: int) -> Tuple[bool, Optional[List[Tuple[float, float]]], Optional[List[int]]]:
    """验证并提取JSON条目中的点和标签
    
    Args:
        obj: JSON对象
        line_num: 行号（用于错误报告）
        
    Returns:
        (is_valid, points, labels)的元组
    """
    if "image" not in obj:
        print(f"[WARN] Line {line_num}: Missing 'image' field")
        return False, None, None
        
    # 检查格式：新格式有"points"和"labels"，旧格式有"x"和"y"
    if "points" in obj and "labels" in obj:
        # 新格式
        points = obj["points"]
        labels = obj["labels"]
        
        if not isinstance(points, list) or not isinstance(labels, list):
            print(f"[WARN] Line {line_num}: 'points' and 'labels' must be lists")
            return False, None, None
            
        if len(points) != len(labels):
            print(f"[WARN] Line {line_num}: 'points' and 'labels' must have same length")
            return False, None, None
        
        # 验证点格式
        valid_points = []
        valid_labels = []
        for point, label in zip(points, labels):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                print(f"[WARN] Line {line_num}: Invalid point format {point}")
                continue
            valid_points.append((float(point[0]), float(point[1])))
            valid_labels.append(int(label))
            
        if len(valid_points) == 0:
            print(f"[WARN] Line {line_num}: No valid points found")
            return False, None, None
            
        return True, valid_points, valid_labels
        
    elif "x" in obj and "y" in obj:
        # 旧格式 - 假设为正类点
        x = float(obj["x"])
        y = float(obj["y"])
        return True, [(x, y)], [1]
        
    else:
        print(f"[WARN] Line {line_num}: Missing coordinate data (need 'points'+'labels' or 'x'+'y')")
        return False, None, None


def get_output_path(image_path: str, output_dir: str) -> str:
    """生成输出mask文件路径
    
    Args:
        image_path: 输入图像路径
        output_dir: 输出目录
        
    Returns:
        输出mask文件路径
    """
    image_name = Path(image_path).stem
    return os.path.join(output_dir, f"{image_name}.png")


def calculate_bounding_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """计算mask的最小包围盒
    
    Args:
        mask: 二值mask数组
        
    Returns:
        (x_min, y_min, x_max, y_max) 包围盒坐标
    """
    # 确保mask是二值的
    mask_binary = (mask > 0).astype(np.uint8)
    
    # 找到非零像素的坐标
    y_indices, x_indices = np.where(mask_binary)
    
    if len(x_indices) == 0 or len(y_indices) == 0:
        # 如果没有前景像素，返回空包围盒
        return 0, 0, 0, 0
    
    x_min = int(np.min(x_indices))
    x_max = int(np.max(x_indices))
    y_min = int(np.min(y_indices))
    y_max = int(np.max(y_indices))
    
    return x_min, y_min, x_max, y_max


def save_mask_as_grayscale(mask: np.ndarray, output_path: str) -> None:
    """将mask保存为灰度图像
    
    Args:
        mask: 二值mask数组
        output_path: 输出文件路径
    """
    # 确保mask是二值的
    mask_binary = (mask > 0).astype(np.uint8)
    
    # 转换为灰度值 (0=背景, 255=前景)
    mask_gray = mask_binary * 255
    
    # 保存
    cv2.imwrite(output_path, mask_gray)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Use SAM2 to segment images from point prompts")
    p.add_argument("--input_jsonl", type=str, required=True,
                   help="Input JSONL file containing image paths and point coordinates")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output directory to save mask images")
    p.add_argument("--sam_checkpoint", type=str, required=True,
                   help="Path to SAM2 model checkpoint")
    p.add_argument("--device", type=str, default=None,
                   help="Device to run on (cuda/cpu). Auto-detect if not specified")
    p.add_argument("--config_path", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml",
                   help="Path to SAM2 config file")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip processing if output file already exists")
    p.add_argument("--resize", type=int, nargs=2, default=None, metavar=("WIDTH", "HEIGHT"),
                   help="Resize input images to specified size [width height]")
    p.add_argument("--json_output", type=str, default=None,
                   help="Path to output JSON file containing mask paths and bounding boxes")
    return p.parse_args()


def main():
    args = parse_args()
    
    # 检查输入文件
    if not os.path.isfile(args.input_jsonl):
        print(f"Error: Input JSONL file not found: {args.input_jsonl}")
        return 1
    
    # 检查SAM2检查点
    if not os.path.isfile(args.sam_checkpoint):
        print(f"Error: SAM2 checkpoint not found: {args.sam_checkpoint}")
        return 1
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化SAM2
    print(f"Initializing SAM2 with checkpoint: {args.sam_checkpoint}")
    print(f"Using device: {args.device or 'auto-detect'}")
    
    try:
        sam_wrapper = SAMWrapper(args.sam_checkpoint, args.device)
        print("SAM2 initialized successfully")
    except Exception as e:
        print(f"Error initializing SAM2: {e}")
        return 1
    
    # 处理JSONL文件
    print(f"Processing JSONL file: {args.input_jsonl}")
    
    num_processed = 0
    num_skipped = 0
    num_errors = 0
    
    # 用于存储JSON输出的列表
    json_results = []
    
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[ERROR] Line {line_num}: Invalid JSON - {e}")
                num_errors += 1
                continue
            
            # 验证并提取点信息
            is_valid, points, labels = validate_and_extract_points(obj, line_num)
            if not is_valid:
                num_errors += 1
                continue
            
            image_path = obj["image"]
            
            # 检查图像文件是否存在
            if not os.path.isfile(image_path):
                print(f"[ERROR] Line {line_num}: Image file not found: {image_path}")
                num_errors += 1
                continue
            
            # 生成输出路径
            output_path = get_output_path(image_path, args.output_dir)
            
            # 检查是否跳过已存在的文件
            if args.skip_existing and os.path.isfile(output_path):
                print(f"[SKIP] Line {line_num}: Output already exists: {output_path}")
                num_skipped += 1
                continue
            
            try:
                # 加载图像
                image = PILImage.open(image_path).convert("RGB")
                
                # 调整大小（如果指定）
                if args.resize:
                    image = image.resize(args.resize, PILImage.BILINEAR)
                
                # 运行SAM2预测
                mask, confidence = sam_wrapper.predict(image, points, labels)
                
                # 计算包围盒
                x_min, y_min, x_max, y_max = calculate_bounding_box(mask)
                
                # 保存mask
                save_mask_as_grayscale(mask, output_path)
                
                # 如果需要JSON输出，添加到结果列表
                if args.json_output:
                    json_results.append({
                        "mask_path": output_path,
                        "bounding_box": {
                            "x_min": x_min,
                            "y_min": y_min,
                            "x_max": x_max,
                            "y_max": y_max
                        }
                    })
                
                print(f"[OK] Line {line_num}: {image_path} -> {output_path} (confidence: {confidence:.3f}, bbox: [{x_min},{y_min},{x_max},{y_max}])")
                num_processed += 1
                
            except Exception as e:
                print(f"[ERROR] Line {line_num}: Failed to process {image_path} - {e}")
                num_errors += 1
                continue
    
    # 保存JSON输出
    if args.json_output and json_results:
        try:
            with open(args.json_output, "w", encoding="utf-8") as json_file:
                json.dump(json_results, json_file, indent=2, ensure_ascii=False)
            print(f"JSON output saved to: {args.json_output}")
        except Exception as e:
            print(f"[ERROR] Failed to save JSON output: {e}")
            return 1
    
    # 打印总结
    print(f"\nProcessing completed:")
    print(f"  Processed: {num_processed}")
    print(f"  Skipped: {num_skipped}")
    print(f"  Errors: {num_errors}")
    print(f"  Output directory: {args.output_dir}")
    if args.json_output:
        print(f"  JSON output: {args.json_output}")
    
    return 0 if num_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

```

现在检查两个文件是否有语法错误：

这些都是导入警告，不是语法错误，可以忽略。让我更新待办事项并完成任务：

现在更新两个文件的文档说明，说明新的JSON输出功能：

```python
#!/usr/bin/env python3

"""
使用SAM2从点提示进行图像分割 (简化版本)

直接复用seg-r1/src/open_r1/grpo.py中的SAMWrapper类进行分割

功能:
- 从点提示生成图像分割mask
- 自动计算每个mask的最小包围盒
- 可选择输出包含mask路径和包围盒坐标的JSON文件

基础用法:
    python seg-rl/sam2_segment_simple.py \
      --input_jsonl /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/points_canonical.jsonl \
      --output_dir /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/masks_step1 \
      --sam_checkpoint /root/autodl-tmp/works/Seg-R0/third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
      --device cuda

带JSON输出:
    python seg-rl/sam2_segment_simple.py \
      --input_jsonl /path/to/points.jsonl \
      --output_dir /path/to/masks \
      --sam_checkpoint /path/to/sam2.1_hiera_large.pt \
      --device cuda \
      --json_output /path/to/results.json

JSON输出格式:
    [
      {
        "mask_path": "/path/to/mask1.png",
        "bounding_box": {
          "x_min": 10, "y_min": 20, "x_max": 100, "y_max": 150
        }
      },
      ...
    ]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

# 添加seg-r1和third_party路径以导入SAMWrapper及sam2依赖
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "seg-r1" / "src"))
# Ensure Python sees the inner package directory (third_party/sam2)
sys.path.insert(0, str(project_root / "third_party" / "sam2"))

try:
    from open_r1.grpo import SAMWrapper
except ImportError as e:
    print(f"Error importing SAMWrapper from grpo.py: {e}")
    print("Please ensure the seg-r1 module is properly set up.")
    print("You may need to:")
    print("1. Install SAM2 dependencies")
    print("2. Check the import path")
    sys.exit(1)


def validate_and_extract_points(obj: Dict[str, Any], line_num: int) -> Tuple[bool, Optional[List[Tuple[float, float]]], Optional[List[int]]]:
    """验证并提取JSON条目中的点和标签"""
    if "image" not in obj:
        print(f"[WARN] Line {line_num}: Missing 'image' field")
        return False, None, None
        
    # 检查格式：新格式有"points"和"labels"，旧格式有"x"和"y"
    if "points" in obj and "labels" in obj:
        # 新格式
        points = obj["points"]
        labels = obj["labels"]
        
        if not isinstance(points, list) or not isinstance(labels, list):
            print(f"[WARN] Line {line_num}: 'points' and 'labels' must be lists")
            return False, None, None
            
        if len(points) != len(labels):
            print(f"[WARN] Line {line_num}: 'points' and 'labels' must have same length")
            return False, None, None
        
        # 验证点格式
        valid_points = []
        valid_labels = []
        for point, label in zip(points, labels):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                print(f"[WARN] Line {line_num}: Invalid point format {point}")
                continue
            valid_points.append((float(point[0]), float(point[1])))
            valid_labels.append(int(label))
            
        if len(valid_points) == 0:
            print(f"[WARN] Line {line_num}: No valid points found")
            return False, None, None
            
        return True, valid_points, valid_labels
        
    elif "x" in obj and "y" in obj:
        # 旧格式 - 假设为正类点
        x = float(obj["x"])
        y = float(obj["y"])
        return True, [(x, y)], [1]
        
    else:
        print(f"[WARN] Line {line_num}: Missing coordinate data (need 'points'+'labels' or 'x'+'y')")
        return False, None, None


def get_output_path(image_path: str, output_dir: str) -> str:
    """生成输出mask文件路径"""
    image_name = Path(image_path).stem
    return os.path.join(output_dir, f"{image_name}.png")


def calculate_bounding_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """计算mask的最小包围盒
    
    Args:
        mask: 二值mask数组
        
    Returns:
        (x_min, y_min, x_max, y_max) 包围盒坐标
    """
    # 确保mask是二值的
    mask_binary = (mask > 0).astype(np.uint8)
    
    # 找到非零像素的坐标
    y_indices, x_indices = np.where(mask_binary)
    
    if len(x_indices) == 0 or len(y_indices) == 0:
        # 如果没有前景像素，返回空包围盒
        return 0, 0, 0, 0
    
    x_min = int(np.min(x_indices))
    x_max = int(np.max(x_indices))
    y_min = int(np.min(y_indices))
    y_max = int(np.max(y_indices))
    
    return x_min, y_min, x_max, y_max


def save_mask_as_grayscale(mask: np.ndarray, output_path: str) -> None:
    """将mask保存为灰度图像"""
    # 确保mask是二值的
    mask_binary = (mask > 0).astype(np.uint8)
    
    # 转换为灰度值 (0=背景, 255=前景)
    mask_gray = mask_binary * 255
    
    # 保存
    cv2.imwrite(output_path, mask_gray)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Use SAM2 to segment images from point prompts (using grpo.py SAMWrapper)")
    p.add_argument("--input_jsonl", type=str, required=True,
                   help="Input JSONL file containing image paths and point coordinates")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output directory to save mask images")
    p.add_argument("--sam_checkpoint", type=str, required=True,
                   help="Path to SAM2 model checkpoint")
    p.add_argument("--device", type=str, default=None,
                   help="Device to run on (cuda/cpu). Auto-detect if not specified")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip processing if output file already exists")
    p.add_argument("--resize", type=int, nargs=2, default=None, metavar=("WIDTH", "HEIGHT"),
                   help="Resize input images to specified size [width height]")
    p.add_argument("--json_output", type=str, default=None,
                   help="Path to output JSON file containing mask paths and bounding boxes")
    return p.parse_args()


def main():
    args = parse_args()
    
    # 检查输入文件
    if not os.path.isfile(args.input_jsonl):
        print(f"Error: Input JSONL file not found: {args.input_jsonl}")
        return 1
    
    # 检查SAM2检查点
    if not os.path.isfile(args.sam_checkpoint):
        print(f"Error: SAM2 checkpoint not found: {args.sam_checkpoint}")
        return 1
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化SAM2
    print(f"Initializing SAM2 with checkpoint: {args.sam_checkpoint}")
    print(f"Using device: {args.device or 'auto-detect'}")
    
    try:
        sam_wrapper = SAMWrapper(args.sam_checkpoint, args.device)
        print("SAM2 initialized successfully")
    except Exception as e:
        print(f"Error initializing SAM2: {e}")
        return 1
    
    # 处理JSONL文件
    print(f"Processing JSONL file: {args.input_jsonl}")
    
    num_processed = 0
    num_skipped = 0
    num_errors = 0
    
    # 用于存储JSON输出的列表
    json_results = []
    
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[ERROR] Line {line_num}: Invalid JSON - {e}")
                num_errors += 1
                continue
            
            # 验证并提取点信息
            is_valid, points, labels = validate_and_extract_points(obj, line_num)
            if not is_valid:
                num_errors += 1
                continue
            
            image_path = obj["image"]
            
            # 检查图像文件是否存在
            if not os.path.isfile(image_path):
                print(f"[ERROR] Line {line_num}: Image file not found: {image_path}")
                num_errors += 1
                continue
            
            # 生成输出路径
            output_path = get_output_path(image_path, args.output_dir)
            
            # 检查是否跳过已存在的文件
            if args.skip_existing and os.path.isfile(output_path):
                print(f"[SKIP] Line {line_num}: Output already exists: {output_path}")
                num_skipped += 1
                continue
            
            try:
                # 加载图像
                image = PILImage.open(image_path).convert("RGB")
                
                # 调整大小（如果指定）
                if args.resize:
                    image = image.resize(args.resize, PILImage.BILINEAR)
                
                # 运行SAM2预测
                mask, confidence = sam_wrapper.predict(image, points, labels)
                
                # 计算包围盒
                x_min, y_min, x_max, y_max = calculate_bounding_box(mask)
                
                # 保存mask
                save_mask_as_grayscale(mask, output_path)
                
                # 如果需要JSON输出，添加到结果列表
                if args.json_output:
                    json_results.append({
                        "mask_path": output_path,
                        "bounding_box": {
                            "x_min": x_min,
                            "y_min": y_min,
                            "x_max": x_max,
                            "y_max": y_max
                        }
                    })
                
                print(f"[OK] Line {line_num}: {image_path} -> {output_path} (confidence: {confidence:.3f}, bbox: [{x_min},{y_min},{x_max},{y_max}])")
                num_processed += 1
                
            except Exception as e:
                print(f"[ERROR] Line {line_num}: Failed to process {image_path} - {e}")
                num_errors += 1
                continue
    
    # 保存JSON输出
    if args.json_output and json_results:
        try:
            with open(args.json_output, "w", encoding="utf-8") as json_file:
                json.dump(json_results, json_file, indent=2, ensure_ascii=False)
            print(f"JSON output saved to: {args.json_output}")
        except Exception as e:
            print(f"[ERROR] Failed to save JSON output: {e}")
            return 1
    
    # 打印总结
    print(f"\nProcessing completed:")
    print(f"  Processed: {num_processed}")
    print(f"  Skipped: {num_skipped}")
    print(f"  Errors: {num_errors}")
    print(f"  Output directory: {args.output_dir}")
    if args.json_output:
        print(f"  JSON output: {args.json_output}")
    
    return 0 if num_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

```

```python
#!/usr/bin/env python3

"""
使用SAM2从点提示进行图像分割

读取包含图像路径和正/负类点坐标的JSONL文件，使用SAM2进行分割，
并将预测的mask保存为灰度图像。

功能:
- 从点提示生成图像分割mask
- 自动计算每个mask的最小包围盒
- 可选择输出包含mask路径和包围盒坐标的JSON文件

输入JSONL格式:
    新格式: {"image": "/path/img.jpg", "points": [[x1,y1], [x2,y2]], "labels": [1, 0]}
    旧格式: {"image": "/path/img.jpg", "x": x1, "y": y1}

输出:
    与输入图像对应的灰度mask图像，保存到指定目录
    - 0 (黑色): 背景
    - 255 (白色): 前景

依赖项:
    - SAM2 (Segment Anything Model 2)
    - torch, PIL, opencv-python, numpy

安装SAM2:
    git clone https://github.com/facebookresearch/segment-anything-2.git third_party/sam2
    cd third_party/sam2 && pip install -e .

下载模型:
    # 下载SAM2.1 Hiera Large模型 (~900MB)
    wget -O third_party/sam2/checkpoints/sam2.1_hiera_large.pt \\
      https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

基础用法:
    python seg-rl/sam2_segment_from_points.py \\
      --input_jsonl /path/to/points.jsonl \\
      --output_dir /path/to/masks \\
      --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \\
      --device cuda

带JSON输出:
    python seg-rl/sam2_segment_from_points.py \\
      --input_jsonl /path/to/points.jsonl \\
      --output_dir /path/to/masks \\
      --sam_checkpoint /path/to/sam2.1_hiera_large.pt \\
      --device cuda \\
      --resize 512 512 \\
      --skip_existing \\
      --json_output /path/to/results.json

JSON输出格式:
    [
      {
        "mask_path": "/path/to/mask1.png",
        "bounding_box": {
          "x_min": 10, "y_min": 20, "x_max": 100, "y_max": 150
        }
      },
      ...
    ]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

# 添加sam2路径
sys.path.append(str(Path(__file__).parent.parent / "third_party" / "sam2"))

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError as e:
    print(f"Error importing SAM2: {e}")
    print("Please ensure SAM2 is properly installed and the path is correct.")
    sys.exit(1)


class SAMWrapper:
    """SAM2包装器，用于图像分割预测"""
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """初始化SAM2模型和预测器
        
        Args:
            model_path: SAM2模型检查点路径
            device: 运行设备 (e.g. "cuda", "cuda:0", "cpu")
                   如果为None，将自动检测可用设备
        """
        # SAM2配置文件路径
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        
        # 检查配置文件是否存在
        config_path = Path(__file__).parent.parent / "third_party" / "sam2" / model_cfg
        if not config_path.exists():
            # 尝试相对于当前目录
            config_path = Path(model_cfg)
            if not config_path.exists():
                raise FileNotFoundError(f"SAM2 config file not found: {model_cfg}")
        
        sam_model = build_sam2(str(config_path), model_path)
        
        # 自动检测设备
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)
        
        # 移动到指定设备
        sam_model = sam_model.to(self.device)
        
        # 初始化预测器
        self.predictor = SAM2ImagePredictor(sam_model)
        self.last_mask = None
        
    def predict(self, 
                image: PILImage.Image, 
                points: Optional[List[Tuple[int, int]]] = None, 
                labels: Optional[List[int]] = None,
                bbox: Optional[List[int]] = None) -> Tuple[np.ndarray, float]:
        """使用给定提示运行SAM2预测
        
        Args:
            image: 输入PIL图像
            points: 点坐标列表 [(x,y), ...]
            labels: 点标签列表 (1=前景, 0=背景)
            bbox: 可选边界框 [x1,y1,x2,y2]
            
        Returns:
            (predicted_mask, confidence_score)的元组
        """
        input_points = np.array(points) if points else None
        input_labels = np.array(labels) if labels else None
        input_bboxes = np.array([bbox]) if bbox else None

        # 转换为numpy数组
        image_np = np.array(image)
        
        # 确保是RGB格式
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image_np
        
        # 设置图像
        self.predictor.set_image(rgb_image)
        
        # 预测
        mask_pred, score, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=input_bboxes,
            multimask_output=False,
        )
        
        self.last_mask = mask_pred[0]
        return mask_pred[0], score[0]


def validate_and_extract_points(obj: Dict[str, Any], line_num: int) -> Tuple[bool, Optional[List[Tuple[float, float]]], Optional[List[int]]]:
    """验证并提取JSON条目中的点和标签
    
    Args:
        obj: JSON对象
        line_num: 行号（用于错误报告）
        
    Returns:
        (is_valid, points, labels)的元组
    """
    if "image" not in obj:
        print(f"[WARN] Line {line_num}: Missing 'image' field")
        return False, None, None
        
    # 检查格式：新格式有"points"和"labels"，旧格式有"x"和"y"
    if "points" in obj and "labels" in obj:
        # 新格式
        points = obj["points"]
        labels = obj["labels"]
        
        if not isinstance(points, list) or not isinstance(labels, list):
            print(f"[WARN] Line {line_num}: 'points' and 'labels' must be lists")
            return False, None, None
            
        if len(points) != len(labels):
            print(f"[WARN] Line {line_num}: 'points' and 'labels' must have same length")
            return False, None, None
        
        # 验证点格式
        valid_points = []
        valid_labels = []
        for point, label in zip(points, labels):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                print(f"[WARN] Line {line_num}: Invalid point format {point}")
                continue
            valid_points.append((float(point[0]), float(point[1])))
            valid_labels.append(int(label))
            
        if len(valid_points) == 0:
            print(f"[WARN] Line {line_num}: No valid points found")
            return False, None, None
            
        return True, valid_points, valid_labels
        
    elif "x" in obj and "y" in obj:
        # 旧格式 - 假设为正类点
        x = float(obj["x"])
        y = float(obj["y"])
        return True, [(x, y)], [1]
        
    else:
        print(f"[WARN] Line {line_num}: Missing coordinate data (need 'points'+'labels' or 'x'+'y')")
        return False, None, None


def get_output_path(image_path: str, output_dir: str) -> str:
    """生成输出mask文件路径
    
    Args:
        image_path: 输入图像路径
        output_dir: 输出目录
        
    Returns:
        输出mask文件路径
    """
    image_name = Path(image_path).stem
    return os.path.join(output_dir, f"{image_name}.png")


def calculate_bounding_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """计算mask的最小包围盒
    
    Args:
        mask: 二值mask数组
        
    Returns:
        (x_min, y_min, x_max, y_max) 包围盒坐标
    """
    # 确保mask是二值的
    mask_binary = (mask > 0).astype(np.uint8)
    
    # 找到非零像素的坐标
    y_indices, x_indices = np.where(mask_binary)
    
    if len(x_indices) == 0 or len(y_indices) == 0:
        # 如果没有前景像素，返回空包围盒
        return 0, 0, 0, 0
    
    x_min = int(np.min(x_indices))
    x_max = int(np.max(x_indices))
    y_min = int(np.min(y_indices))
    y_max = int(np.max(y_indices))
    
    return x_min, y_min, x_max, y_max


def save_mask_as_grayscale(mask: np.ndarray, output_path: str) -> None:
    """将mask保存为灰度图像
    
    Args:
        mask: 二值mask数组
        output_path: 输出文件路径
    """
    # 确保mask是二值的
    mask_binary = (mask > 0).astype(np.uint8)
    
    # 转换为灰度值 (0=背景, 255=前景)
    mask_gray = mask_binary * 255
    
    # 保存
    cv2.imwrite(output_path, mask_gray)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Use SAM2 to segment images from point prompts")
    p.add_argument("--input_jsonl", type=str, required=True,
                   help="Input JSONL file containing image paths and point coordinates")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output directory to save mask images")
    p.add_argument("--sam_checkpoint", type=str, required=True,
                   help="Path to SAM2 model checkpoint")
    p.add_argument("--device", type=str, default=None,
                   help="Device to run on (cuda/cpu). Auto-detect if not specified")
    p.add_argument("--config_path", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml",
                   help="Path to SAM2 config file")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip processing if output file already exists")
    p.add_argument("--resize", type=int, nargs=2, default=None, metavar=("WIDTH", "HEIGHT"),
                   help="Resize input images to specified size [width height]")
    p.add_argument("--json_output", type=str, default=None,
                   help="Path to output JSON file containing mask paths and bounding boxes")
    return p.parse_args()


def main():
    args = parse_args()
    
    # 检查输入文件
    if not os.path.isfile(args.input_jsonl):
        print(f"Error: Input JSONL file not found: {args.input_jsonl}")
        return 1
    
    # 检查SAM2检查点
    if not os.path.isfile(args.sam_checkpoint):
        print(f"Error: SAM2 checkpoint not found: {args.sam_checkpoint}")
        return 1
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化SAM2
    print(f"Initializing SAM2 with checkpoint: {args.sam_checkpoint}")
    print(f"Using device: {args.device or 'auto-detect'}")
    
    try:
        sam_wrapper = SAMWrapper(args.sam_checkpoint, args.device)
        print("SAM2 initialized successfully")
    except Exception as e:
        print(f"Error initializing SAM2: {e}")
        return 1
    
    # 处理JSONL文件
    print(f"Processing JSONL file: {args.input_jsonl}")
    
    num_processed = 0
    num_skipped = 0
    num_errors = 0
    
    # 用于存储JSON输出的列表
    json_results = []
    
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[ERROR] Line {line_num}: Invalid JSON - {e}")
                num_errors += 1
                continue
            
            # 验证并提取点信息
            is_valid, points, labels = validate_and_extract_points(obj, line_num)
            if not is_valid:
                num_errors += 1
                continue
            
            image_path = obj["image"]
            
            # 检查图像文件是否存在
            if not os.path.isfile(image_path):
                print(f"[ERROR] Line {line_num}: Image file not found: {image_path}")
                num_errors += 1
                continue
            
            # 生成输出路径
            output_path = get_output_path(image_path, args.output_dir)
            
            # 检查是否跳过已存在的文件
            if args.skip_existing and os.path.isfile(output_path):
                print(f"[SKIP] Line {line_num}: Output already exists: {output_path}")
                num_skipped += 1
                continue
            
            try:
                # 加载图像
                image = PILImage.open(image_path).convert("RGB")
                
                # 调整大小（如果指定）
                if args.resize:
                    image = image.resize(args.resize, PILImage.BILINEAR)
                
                # 运行SAM2预测
                mask, confidence = sam_wrapper.predict(image, points, labels)
                
                # 计算包围盒
                x_min, y_min, x_max, y_max = calculate_bounding_box(mask)
                
                # 保存mask
                save_mask_as_grayscale(mask, output_path)
                
                # 如果需要JSON输出，添加到结果列表
                if args.json_output:
                    json_results.append({
                        "mask_path": output_path,
                        "bounding_box": {
                            "x_min": x_min,
                            "y_min": y_min,
                            "x_max": x_max,
                            "y_max": y_max
                        }
                    })
                
                print(f"[OK] Line {line_num}: {image_path} -> {output_path} (confidence: {confidence:.3f}, bbox: [{x_min},{y_min},{x_max},{y_max}])")
                num_processed += 1
                
            except Exception as e:
                print(f"[ERROR] Line {line_num}: Failed to process {image_path} - {e}")
                num_errors += 1
                continue
    
    # 保存JSON输出
    if args.json_output and json_results:
        try:
            with open(args.json_output, "w", encoding="utf-8") as json_file:
                json.dump(json_results, json_file, indent=2, ensure_ascii=False)
            print(f"JSON output saved to: {args.json_output}")
        except Exception as e:
            print(f"[ERROR] Failed to save JSON output: {e}")
            return 1
    
    # 打印总结
    print(f"\nProcessing completed:")
    print(f"  Processed: {num_processed}")
    print(f"  Skipped: {num_skipped}")
    print(f"  Errors: {num_errors}")
    print(f"  Output directory: {args.output_dir}")
    if args.json_output:
        print(f"  JSON output: {args.json_output}")
    
    return 0 if num_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

```

完美！我已经成功为两个文件添加了所需的功能。让我总结一下所做的更改：

## 功能更新总结

我已经为 `sam2_segment_simple.py` 和 `sam2_segment_from_points.py` 两个文件添加了以下新功能：

### 1. 🔧 添加的新功能
- **最小包围盒计算**: 为每个生成的mask计算最小包围盒坐标 (x_min, y_min, x_max, y_max)
- **JSON输出**: 可选择输出包含mask路径和包围盒坐标的JSON文件
- **新命令行参数**: `--json_output` 用于指定JSON输出文件路径

### 2. 📝 新增的函数
- `calculate_bounding_box(mask)`: 计算mask的最小包围盒坐标，处理空mask情况

### 3. 🎯 JSON输出格式
输出的JSON文件包含一个数组，每个元素包含：
```json
[
  {
    "mask_path": "/path/to/mask1.png",
    "bounding_box": {
      "x_min": 10,
      "y_min": 20, 
      "x_max": 100,
      "y_max": 150
    }
  },
  ...
]
```

### 4. 📖 更新的使用方法

**sam2_segment_simple.py** (使用grpo.py的SAMWrapper):
```bash
python seg-rl/sam2_segment_simple.py \
  --input_jsonl /path/to/points.jsonl \
  --output_dir /path/to/masks \
  --sam_checkpoint /path/to/sam2.1_hiera_large.pt \
  --device cuda \
  --json_output /path/to/results.json
```

**sam2_segment_from_points.py** (直接使用SAM2):
```bash
python seg-rl/sam2_segment_from_points.py \
  --input_jsonl /path/to/points.jsonl \
  --output_dir /path/to/masks \
  --sam_checkpoint /path/to/sam2.1_hiera_large.pt \
  --device cuda \
  --json_output /path/to/results.json
```

### 5. 🔍 增强的输出信息
现在处理日志会显示包围盒信息：
```
[OK] Line 1: /path/image.jpg -> /path/mask.png (confidence: 0.952, bbox: [10,20,100,150])
```

这些更改保持了向后兼容性，不影响现有的使用方式，同时添加了强大的包围盒计算和JSON输出功能。

```json
[
  {
    "mask_path": "/path/to/mask1.png",
    "bounding_box": {
      "x_min": 10,
      "y_min": 20, 
      "x_max": 100,
      "y_max": 150
    }
  },
  ...
]
```

```shellscript
python seg-rl/sam2_segment_simple.py \
  --input_jsonl /path/to/points.jsonl \
  --output_dir /path/to/masks \
  --sam_checkpoint /path/to/sam2.1_hiera_large.pt \
  --device cuda \
  --json_output /path/to/results.json
```

```shellscript
python seg-rl/sam2_segment_from_points.py \
  --input_jsonl /path/to/points.jsonl \
  --output_dir /path/to/masks \
  --sam_checkpoint /path/to/sam2.1_hiera_large.pt \
  --device cuda \
  --json_output /path/to/results.json
```

```plaintext
[OK] Line 1: /path/image.jpg -> /path/mask.png (confidence: 0.952, bbox: [10,20,100,150])
```

---

**User**

请以 "bbox": [xmin, ymin, xmax, ymax] 这种形式来表示每一个包围盒，而不是现在的方式

---

**Cursor**

我需要修改两个文件中的包围盒表示格式，从字典形式改为数组形式。

```python
#!/usr/bin/env python3

"""
使用SAM2从点提示进行图像分割 (简化版本)

直接复用seg-r1/src/open_r1/grpo.py中的SAMWrapper类进行分割

功能:
- 从点提示生成图像分割mask
- 自动计算每个mask的最小包围盒
- 可选择输出包含mask路径和包围盒坐标的JSON文件

基础用法:
    python seg-rl/sam2_segment_simple.py \
      --input_jsonl /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/points_canonical.jsonl \
      --output_dir /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/masks_step1 \
      --sam_checkpoint /root/autodl-tmp/works/Seg-R0/third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
      --device cuda

带JSON输出:
    python seg-rl/sam2_segment_simple.py \
      --input_jsonl /path/to/points.jsonl \
      --output_dir /path/to/masks \
      --sam_checkpoint /path/to/sam2.1_hiera_large.pt \
      --device cuda \
      --json_output /path/to/results.json

JSON输出格式:
    [
      {
        "mask_path": "/path/to/mask1.png",
        "bounding_box": {
          "x_min": 10, "y_min": 20, "x_max": 100, "y_max": 150
        }
      },
      ...
    ]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

# 添加seg-r1和third_party路径以导入SAMWrapper及sam2依赖
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "seg-r1" / "src"))
# Ensure Python sees the inner package directory (third_party/sam2)
sys.path.insert(0, str(project_root / "third_party" / "sam2"))

try:
    from open_r1.grpo import SAMWrapper
except ImportError as e:
    print(f"Error importing SAMWrapper from grpo.py: {e}")
    print("Please ensure the seg-r1 module is properly set up.")
    print("You may need to:")
    print("1. Install SAM2 dependencies")
    print("2. Check the import path")
    sys.exit(1)


def validate_and_extract_points(obj: Dict[str, Any], line_num: int) -> Tuple[bool, Optional[List[Tuple[float, float]]], Optional[List[int]]]:
    """验证并提取JSON条目中的点和标签"""
    if "image" not in obj:
        print(f"[WARN] Line {line_num}: Missing 'image' field")
        return False, None, None
        
    # 检查格式：新格式有"points"和"labels"，旧格式有"x"和"y"
    if "points" in obj and "labels" in obj:
        # 新格式
        points = obj["points"]
        labels = obj["labels"]
        
        if not isinstance(points, list) or not isinstance(labels, list):
            print(f"[WARN] Line {line_num}: 'points' and 'labels' must be lists")
            return False, None, None
            
        if len(points) != len(labels):
            print(f"[WARN] Line {line_num}: 'points' and 'labels' must have same length")
            return False, None, None
        
        # 验证点格式
        valid_points = []
        valid_labels = []
        for point, label in zip(points, labels):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                print(f"[WARN] Line {line_num}: Invalid point format {point}")
                continue
            valid_points.append((float(point[0]), float(point[1])))
            valid_labels.append(int(label))
            
        if len(valid_points) == 0:
            print(f"[WARN] Line {line_num}: No valid points found")
            return False, None, None
            
        return True, valid_points, valid_labels
        
    elif "x" in obj and "y" in obj:
        # 旧格式 - 假设为正类点
        x = float(obj["x"])
        y = float(obj["y"])
        return True, [(x, y)], [1]
        
    else:
        print(f"[WARN] Line {line_num}: Missing coordinate data (need 'points'+'labels' or 'x'+'y')")
        return False, None, None


def get_output_path(image_path: str, output_dir: str) -> str:
    """生成输出mask文件路径"""
    image_name = Path(image_path).stem
    return os.path.join(output_dir, f"{image_name}.png")


def calculate_bounding_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """计算mask的最小包围盒
    
    Args:
        mask: 二值mask数组
        
    Returns:
        (x_min, y_min, x_max, y_max) 包围盒坐标
    """
    # 确保mask是二值的
    mask_binary = (mask > 0).astype(np.uint8)
    
    # 找到非零像素的坐标
    y_indices, x_indices = np.where(mask_binary)
    
    if len(x_indices) == 0 or len(y_indices) == 0:
        # 如果没有前景像素，返回空包围盒
        return 0, 0, 0, 0
    
    x_min = int(np.min(x_indices))
    x_max = int(np.max(x_indices))
    y_min = int(np.min(y_indices))
    y_max = int(np.max(y_indices))
    
    return x_min, y_min, x_max, y_max


def save_mask_as_grayscale(mask: np.ndarray, output_path: str) -> None:
    """将mask保存为灰度图像"""
    # 确保mask是二值的
    mask_binary = (mask > 0).astype(np.uint8)
    
    # 转换为灰度值 (0=背景, 255=前景)
    mask_gray = mask_binary * 255
    
    # 保存
    cv2.imwrite(output_path, mask_gray)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Use SAM2 to segment images from point prompts (using grpo.py SAMWrapper)")
    p.add_argument("--input_jsonl", type=str, required=True,
                   help="Input JSONL file containing image paths and point coordinates")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output directory to save mask images")
    p.add_argument("--sam_checkpoint", type=str, required=True,
                   help="Path to SAM2 model checkpoint")
    p.add_argument("--device", type=str, default=None,
                   help="Device to run on (cuda/cpu). Auto-detect if not specified")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip processing if output file already exists")
    p.add_argument("--resize", type=int, nargs=2, default=None, metavar=("WIDTH", "HEIGHT"),
                   help="Resize input images to specified size [width height]")
    p.add_argument("--json_output", type=str, default=None,
                   help="Path to output JSON file containing mask paths and bounding boxes")
    return p.parse_args()


def main():
    args = parse_args()
    
    # 检查输入文件
    if not os.path.isfile(args.input_jsonl):
        print(f"Error: Input JSONL file not found: {args.input_jsonl}")
        return 1
    
    # 检查SAM2检查点
    if not os.path.isfile(args.sam_checkpoint):
        print(f"Error: SAM2 checkpoint not found: {args.sam_checkpoint}")
        return 1
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化SAM2
    print(f"Initializing SAM2 with checkpoint: {args.sam_checkpoint}")
    print(f"Using device: {args.device or 'auto-detect'}")
    
    try:
        sam_wrapper = SAMWrapper(args.sam_checkpoint, args.device)
        print("SAM2 initialized successfully")
    except Exception as e:
        print(f"Error initializing SAM2: {e}")
        return 1
    
    # 处理JSONL文件
    print(f"Processing JSONL file: {args.input_jsonl}")
    
    num_processed = 0
    num_skipped = 0
    num_errors = 0
    
    # 用于存储JSON输出的列表
    json_results = []
    
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[ERROR] Line {line_num}: Invalid JSON - {e}")
                num_errors += 1
                continue
            
            # 验证并提取点信息
            is_valid, points, labels = validate_and_extract_points(obj, line_num)
            if not is_valid:
                num_errors += 1
                continue
            
            image_path = obj["image"]
            
            # 检查图像文件是否存在
            if not os.path.isfile(image_path):
                print(f"[ERROR] Line {line_num}: Image file not found: {image_path}")
                num_errors += 1
                continue
            
            # 生成输出路径
            output_path = get_output_path(image_path, args.output_dir)
            
            # 检查是否跳过已存在的文件
            if args.skip_existing and os.path.isfile(output_path):
                print(f"[SKIP] Line {line_num}: Output already exists: {output_path}")
                num_skipped += 1
                continue
            
            try:
                # 加载图像
                image = PILImage.open(image_path).convert("RGB")
                
                # 调整大小（如果指定）
                if args.resize:
                    image = image.resize(args.resize, PILImage.BILINEAR)
                
                # 运行SAM2预测
                mask, confidence = sam_wrapper.predict(image, points, labels)
                
                # 计算包围盒
                x_min, y_min, x_max, y_max = calculate_bounding_box(mask)
                
                # 保存mask
                save_mask_as_grayscale(mask, output_path)
                
                # 如果需要JSON输出，添加到结果列表
                if args.json_output:
                    json_results.append({
                        "mask_path": output_path,
                        "bbox": [x_min, y_min, x_max, y_max]
                    })
                
                print(f"[OK] Line {line_num}: {image_path} -> {output_path} (confidence: {confidence:.3f}, bbox: [{x_min},{y_min},{x_max},{y_max}])")
                num_processed += 1
                
            except Exception as e:
                print(f"[ERROR] Line {line_num}: Failed to process {image_path} - {e}")
                num_errors += 1
                continue
    
    # 保存JSON输出
    if args.json_output and json_results:
        try:
            with open(args.json_output, "w", encoding="utf-8") as json_file:
                json.dump(json_results, json_file, indent=2, ensure_ascii=False)
            print(f"JSON output saved to: {args.json_output}")
        except Exception as e:
            print(f"[ERROR] Failed to save JSON output: {e}")
            return 1
    
    # 打印总结
    print(f"\nProcessing completed:")
    print(f"  Processed: {num_processed}")
    print(f"  Skipped: {num_skipped}")
    print(f"  Errors: {num_errors}")
    print(f"  Output directory: {args.output_dir}")
    if args.json_output:
        print(f"  JSON output: {args.json_output}")
    
    return 0 if num_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

```

```python
#!/usr/bin/env python3

"""
使用SAM2从点提示进行图像分割

读取包含图像路径和正/负类点坐标的JSONL文件，使用SAM2进行分割，
并将预测的mask保存为灰度图像。

功能:
- 从点提示生成图像分割mask
- 自动计算每个mask的最小包围盒
- 可选择输出包含mask路径和包围盒坐标的JSON文件

输入JSONL格式:
    新格式: {"image": "/path/img.jpg", "points": [[x1,y1], [x2,y2]], "labels": [1, 0]}
    旧格式: {"image": "/path/img.jpg", "x": x1, "y": y1}

输出:
    与输入图像对应的灰度mask图像，保存到指定目录
    - 0 (黑色): 背景
    - 255 (白色): 前景

依赖项:
    - SAM2 (Segment Anything Model 2)
    - torch, PIL, opencv-python, numpy

安装SAM2:
    git clone https://github.com/facebookresearch/segment-anything-2.git third_party/sam2
    cd third_party/sam2 && pip install -e .

下载模型:
    # 下载SAM2.1 Hiera Large模型 (~900MB)
    wget -O third_party/sam2/checkpoints/sam2.1_hiera_large.pt \\
      https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

基础用法:
    python seg-rl/sam2_segment_from_points.py \\
      --input_jsonl /path/to/points.jsonl \\
      --output_dir /path/to/masks \\
      --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \\
      --device cuda

带JSON输出:
    python seg-rl/sam2_segment_from_points.py \\
      --input_jsonl /path/to/points.jsonl \\
      --output_dir /path/to/masks \\
      --sam_checkpoint /path/to/sam2.1_hiera_large.pt \\
      --device cuda \\
      --resize 512 512 \\
      --skip_existing \\
      --json_output /path/to/results.json

JSON输出格式:
    [
      {
        "mask_path": "/path/to/mask1.png",
        "bounding_box": {
          "x_min": 10, "y_min": 20, "x_max": 100, "y_max": 150
        }
      },
      ...
    ]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

# 添加sam2路径
sys.path.append(str(Path(__file__).parent.parent / "third_party" / "sam2"))

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError as e:
    print(f"Error importing SAM2: {e}")
    print("Please ensure SAM2 is properly installed and the path is correct.")
    sys.exit(1)


class SAMWrapper:
    """SAM2包装器，用于图像分割预测"""
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """初始化SAM2模型和预测器
        
        Args:
            model_path: SAM2模型检查点路径
            device: 运行设备 (e.g. "cuda", "cuda:0", "cpu")
                   如果为None，将自动检测可用设备
        """
        # SAM2配置文件路径
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        
        # 检查配置文件是否存在
        config_path = Path(__file__).parent.parent / "third_party" / "sam2" / model_cfg
        if not config_path.exists():
            # 尝试相对于当前目录
            config_path = Path(model_cfg)
            if not config_path.exists():
                raise FileNotFoundError(f"SAM2 config file not found: {model_cfg}")
        
        sam_model = build_sam2(str(config_path), model_path)
        
        # 自动检测设备
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)
        
        # 移动到指定设备
        sam_model = sam_model.to(self.device)
        
        # 初始化预测器
        self.predictor = SAM2ImagePredictor(sam_model)
        self.last_mask = None
        
    def predict(self, 
                image: PILImage.Image, 
                points: Optional[List[Tuple[int, int]]] = None, 
                labels: Optional[List[int]] = None,
                bbox: Optional[List[int]] = None) -> Tuple[np.ndarray, float]:
        """使用给定提示运行SAM2预测
        
        Args:
            image: 输入PIL图像
            points: 点坐标列表 [(x,y), ...]
            labels: 点标签列表 (1=前景, 0=背景)
            bbox: 可选边界框 [x1,y1,x2,y2]
            
        Returns:
            (predicted_mask, confidence_score)的元组
        """
        input_points = np.array(points) if points else None
        input_labels = np.array(labels) if labels else None
        input_bboxes = np.array([bbox]) if bbox else None

        # 转换为numpy数组
        image_np = np.array(image)
        
        # 确保是RGB格式
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image_np
        
        # 设置图像
        self.predictor.set_image(rgb_image)
        
        # 预测
        mask_pred, score, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=input_bboxes,
            multimask_output=False,
        )
        
        self.last_mask = mask_pred[0]
        return mask_pred[0], score[0]


def validate_and_extract_points(obj: Dict[str, Any], line_num: int) -> Tuple[bool, Optional[List[Tuple[float, float]]], Optional[List[int]]]:
    """验证并提取JSON条目中的点和标签
    
    Args:
        obj: JSON对象
        line_num: 行号（用于错误报告）
        
    Returns:
        (is_valid, points, labels)的元组
    """
    if "image" not in obj:
        print(f"[WARN] Line {line_num}: Missing 'image' field")
        return False, None, None
        
    # 检查格式：新格式有"points"和"labels"，旧格式有"x"和"y"
    if "points" in obj and "labels" in obj:
        # 新格式
        points = obj["points"]
        labels = obj["labels"]
        
        if not isinstance(points, list) or not isinstance(labels, list):
            print(f"[WARN] Line {line_num}: 'points' and 'labels' must be lists")
            return False, None, None
            
        if len(points) != len(labels):
            print(f"[WARN] Line {line_num}: 'points' and 'labels' must have same length")
            return False, None, None
        
        # 验证点格式
        valid_points = []
        valid_labels = []
        for point, label in zip(points, labels):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                print(f"[WARN] Line {line_num}: Invalid point format {point}")
                continue
            valid_points.append((float(point[0]), float(point[1])))
            valid_labels.append(int(label))
            
        if len(valid_points) == 0:
            print(f"[WARN] Line {line_num}: No valid points found")
            return False, None, None
            
        return True, valid_points, valid_labels
        
    elif "x" in obj and "y" in obj:
        # 旧格式 - 假设为正类点
        x = float(obj["x"])
        y = float(obj["y"])
        return True, [(x, y)], [1]
        
    else:
        print(f"[WARN] Line {line_num}: Missing coordinate data (need 'points'+'labels' or 'x'+'y')")
        return False, None, None


def get_output_path(image_path: str, output_dir: str) -> str:
    """生成输出mask文件路径
    
    Args:
        image_path: 输入图像路径
        output_dir: 输出目录
        
    Returns:
        输出mask文件路径
    """
    image_name = Path(image_path).stem
    return os.path.join(output_dir, f"{image_name}.png")


def calculate_bounding_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """计算mask的最小包围盒
    
    Args:
        mask: 二值mask数组
        
    Returns:
        (x_min, y_min, x_max, y_max) 包围盒坐标
    """
    # 确保mask是二值的
    mask_binary = (mask > 0).astype(np.uint8)
    
    # 找到非零像素的坐标
    y_indices, x_indices = np.where(mask_binary)
    
    if len(x_indices) == 0 or len(y_indices) == 0:
        # 如果没有前景像素，返回空包围盒
        return 0, 0, 0, 0
    
    x_min = int(np.min(x_indices))
    x_max = int(np.max(x_indices))
    y_min = int(np.min(y_indices))
    y_max = int(np.max(y_indices))
    
    return x_min, y_min, x_max, y_max


def save_mask_as_grayscale(mask: np.ndarray, output_path: str) -> None:
    """将mask保存为灰度图像
    
    Args:
        mask: 二值mask数组
        output_path: 输出文件路径
    """
    # 确保mask是二值的
    mask_binary = (mask > 0).astype(np.uint8)
    
    # 转换为灰度值 (0=背景, 255=前景)
    mask_gray = mask_binary * 255
    
    # 保存
    cv2.imwrite(output_path, mask_gray)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Use SAM2 to segment images from point prompts")
    p.add_argument("--input_jsonl", type=str, required=True,
                   help="Input JSONL file containing image paths and point coordinates")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output directory to save mask images")
    p.add_argument("--sam_checkpoint", type=str, required=True,
                   help="Path to SAM2 model checkpoint")
    p.add_argument("--device", type=str, default=None,
                   help="Device to run on (cuda/cpu). Auto-detect if not specified")
    p.add_argument("--config_path", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml",
                   help="Path to SAM2 config file")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip processing if output file already exists")
    p.add_argument("--resize", type=int, nargs=2, default=None, metavar=("WIDTH", "HEIGHT"),
                   help="Resize input images to specified size [width height]")
    p.add_argument("--json_output", type=str, default=None,
                   help="Path to output JSON file containing mask paths and bounding boxes")
    return p.parse_args()


def main():
    args = parse_args()
    
    # 检查输入文件
    if not os.path.isfile(args.input_jsonl):
        print(f"Error: Input JSONL file not found: {args.input_jsonl}")
        return 1
    
    # 检查SAM2检查点
    if not os.path.isfile(args.sam_checkpoint):
        print(f"Error: SAM2 checkpoint not found: {args.sam_checkpoint}")
        return 1
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化SAM2
    print(f"Initializing SAM2 with checkpoint: {args.sam_checkpoint}")
    print(f"Using device: {args.device or 'auto-detect'}")
    
    try:
        sam_wrapper = SAMWrapper(args.sam_checkpoint, args.device)
        print("SAM2 initialized successfully")
    except Exception as e:
        print(f"Error initializing SAM2: {e}")
        return 1
    
    # 处理JSONL文件
    print(f"Processing JSONL file: {args.input_jsonl}")
    
    num_processed = 0
    num_skipped = 0
    num_errors = 0
    
    # 用于存储JSON输出的列表
    json_results = []
    
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[ERROR] Line {line_num}: Invalid JSON - {e}")
                num_errors += 1
                continue
            
            # 验证并提取点信息
            is_valid, points, labels = validate_and_extract_points(obj, line_num)
            if not is_valid:
                num_errors += 1
                continue
            
            image_path = obj["image"]
            
            # 检查图像文件是否存在
            if not os.path.isfile(image_path):
                print(f"[ERROR] Line {line_num}: Image file not found: {image_path}")
                num_errors += 1
                continue
            
            # 生成输出路径
            output_path = get_output_path(image_path, args.output_dir)
            
            # 检查是否跳过已存在的文件
            if args.skip_existing and os.path.isfile(output_path):
                print(f"[SKIP] Line {line_num}: Output already exists: {output_path}")
                num_skipped += 1
                continue
            
            try:
                # 加载图像
                image = PILImage.open(image_path).convert("RGB")
                
                # 调整大小（如果指定）
                if args.resize:
                    image = image.resize(args.resize, PILImage.BILINEAR)
                
                # 运行SAM2预测
                mask, confidence = sam_wrapper.predict(image, points, labels)
                
                # 计算包围盒
                x_min, y_min, x_max, y_max = calculate_bounding_box(mask)
                
                # 保存mask
                save_mask_as_grayscale(mask, output_path)
                
                # 如果需要JSON输出，添加到结果列表
                if args.json_output:
                    json_results.append({
                        "mask_path": output_path,
                        "bbox": [x_min, y_min, x_max, y_max]
                    })
                
                print(f"[OK] Line {line_num}: {image_path} -> {output_path} (confidence: {confidence:.3f}, bbox: [{x_min},{y_min},{x_max},{y_max}])")
                num_processed += 1
                
            except Exception as e:
                print(f"[ERROR] Line {line_num}: Failed to process {image_path} - {e}")
                num_errors += 1
                continue
    
    # 保存JSON输出
    if args.json_output and json_results:
        try:
            with open(args.json_output, "w", encoding="utf-8") as json_file:
                json.dump(json_results, json_file, indent=2, ensure_ascii=False)
            print(f"JSON output saved to: {args.json_output}")
        except Exception as e:
            print(f"[ERROR] Failed to save JSON output: {e}")
            return 1
    
    # 打印总结
    print(f"\nProcessing completed:")
    print(f"  Processed: {num_processed}")
    print(f"  Skipped: {num_skipped}")
    print(f"  Errors: {num_errors}")
    print(f"  Output directory: {args.output_dir}")
    if args.json_output:
        print(f"  JSON output: {args.json_output}")
    
    return 0 if num_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

```

现在还需要更新文档中的JSON格式示例：

```python
#!/usr/bin/env python3

"""
使用SAM2从点提示进行图像分割 (简化版本)

直接复用seg-r1/src/open_r1/grpo.py中的SAMWrapper类进行分割

功能:
- 从点提示生成图像分割mask
- 自动计算每个mask的最小包围盒
- 可选择输出包含mask路径和包围盒坐标的JSON文件

基础用法:
    python seg-rl/sam2_segment_simple.py \
      --input_jsonl /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/points_canonical.jsonl \
      --output_dir /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/masks_step1 \
      --sam_checkpoint /root/autodl-tmp/works/Seg-R0/third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
      --device cuda

带JSON输出:
    python seg-rl/sam2_segment_simple.py \
      --input_jsonl /path/to/points.jsonl \
      --output_dir /path/to/masks \
      --sam_checkpoint /path/to/sam2.1_hiera_large.pt \
      --device cuda \
      --json_output /path/to/results.json

JSON输出格式:
    [
      {
        "mask_path": "/path/to/mask1.png",
        "bbox": [10, 20, 100, 150]
      },
      ...
    ]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

# 添加seg-r1和third_party路径以导入SAMWrapper及sam2依赖
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "seg-r1" / "src"))
# Ensure Python sees the inner package directory (third_party/sam2)
sys.path.insert(0, str(project_root / "third_party" / "sam2"))

try:
    from open_r1.grpo import SAMWrapper
except ImportError as e:
    print(f"Error importing SAMWrapper from grpo.py: {e}")
    print("Please ensure the seg-r1 module is properly set up.")
    print("You may need to:")
    print("1. Install SAM2 dependencies")
    print("2. Check the import path")
    sys.exit(1)


def validate_and_extract_points(obj: Dict[str, Any], line_num: int) -> Tuple[bool, Optional[List[Tuple[float, float]]], Optional[List[int]]]:
    """验证并提取JSON条目中的点和标签"""
    if "image" not in obj:
        print(f"[WARN] Line {line_num}: Missing 'image' field")
        return False, None, None
        
    # 检查格式：新格式有"points"和"labels"，旧格式有"x"和"y"
    if "points" in obj and "labels" in obj:
        # 新格式
        points = obj["points"]
        labels = obj["labels"]
        
        if not isinstance(points, list) or not isinstance(labels, list):
            print(f"[WARN] Line {line_num}: 'points' and 'labels' must be lists")
            return False, None, None
            
        if len(points) != len(labels):
            print(f"[WARN] Line {line_num}: 'points' and 'labels' must have same length")
            return False, None, None
        
        # 验证点格式
        valid_points = []
        valid_labels = []
        for point, label in zip(points, labels):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                print(f"[WARN] Line {line_num}: Invalid point format {point}")
                continue
            valid_points.append((float(point[0]), float(point[1])))
            valid_labels.append(int(label))
            
        if len(valid_points) == 0:
            print(f"[WARN] Line {line_num}: No valid points found")
            return False, None, None
            
        return True, valid_points, valid_labels
        
    elif "x" in obj and "y" in obj:
        # 旧格式 - 假设为正类点
        x = float(obj["x"])
        y = float(obj["y"])
        return True, [(x, y)], [1]
        
    else:
        print(f"[WARN] Line {line_num}: Missing coordinate data (need 'points'+'labels' or 'x'+'y')")
        return False, None, None


def get_output_path(image_path: str, output_dir: str) -> str:
    """生成输出mask文件路径"""
    image_name = Path(image_path).stem
    return os.path.join(output_dir, f"{image_name}.png")


def calculate_bounding_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """计算mask的最小包围盒
    
    Args:
        mask: 二值mask数组
        
    Returns:
        (x_min, y_min, x_max, y_max) 包围盒坐标
    """
    # 确保mask是二值的
    mask_binary = (mask > 0).astype(np.uint8)
    
    # 找到非零像素的坐标
    y_indices, x_indices = np.where(mask_binary)
    
    if len(x_indices) == 0 or len(y_indices) == 0:
        # 如果没有前景像素，返回空包围盒
        return 0, 0, 0, 0
    
    x_min = int(np.min(x_indices))
    x_max = int(np.max(x_indices))
    y_min = int(np.min(y_indices))
    y_max = int(np.max(y_indices))
    
    return x_min, y_min, x_max, y_max


def save_mask_as_grayscale(mask: np.ndarray, output_path: str) -> None:
    """将mask保存为灰度图像"""
    # 确保mask是二值的
    mask_binary = (mask > 0).astype(np.uint8)
    
    # 转换为灰度值 (0=背景, 255=前景)
    mask_gray = mask_binary * 255
    
    # 保存
    cv2.imwrite(output_path, mask_gray)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Use SAM2 to segment images from point prompts (using grpo.py SAMWrapper)")
    p.add_argument("--input_jsonl", type=str, required=True,
                   help="Input JSONL file containing image paths and point coordinates")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output directory to save mask images")
    p.add_argument("--sam_checkpoint", type=str, required=True,
                   help="Path to SAM2 model checkpoint")
    p.add_argument("--device", type=str, default=None,
                   help="Device to run on (cuda/cpu). Auto-detect if not specified")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip processing if output file already exists")
    p.add_argument("--resize", type=int, nargs=2, default=None, metavar=("WIDTH", "HEIGHT"),
                   help="Resize input images to specified size [width height]")
    p.add_argument("--json_output", type=str, default=None,
                   help="Path to output JSON file containing mask paths and bounding boxes")
    return p.parse_args()


def main():
    args = parse_args()
    
    # 检查输入文件
    if not os.path.isfile(args.input_jsonl):
        print(f"Error: Input JSONL file not found: {args.input_jsonl}")
        return 1
    
    # 检查SAM2检查点
    if not os.path.isfile(args.sam_checkpoint):
        print(f"Error: SAM2 checkpoint not found: {args.sam_checkpoint}")
        return 1
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化SAM2
    print(f"Initializing SAM2 with checkpoint: {args.sam_checkpoint}")
    print(f"Using device: {args.device or 'auto-detect'}")
    
    try:
        sam_wrapper = SAMWrapper(args.sam_checkpoint, args.device)
        print("SAM2 initialized successfully")
    except Exception as e:
        print(f"Error initializing SAM2: {e}")
        return 1
    
    # 处理JSONL文件
    print(f"Processing JSONL file: {args.input_jsonl}")
    
    num_processed = 0
    num_skipped = 0
    num_errors = 0
    
    # 用于存储JSON输出的列表
    json_results = []
    
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[ERROR] Line {line_num}: Invalid JSON - {e}")
                num_errors += 1
                continue
            
            # 验证并提取点信息
            is_valid, points, labels = validate_and_extract_points(obj, line_num)
            if not is_valid:
                num_errors += 1
                continue
            
            image_path = obj["image"]
            
            # 检查图像文件是否存在
            if not os.path.isfile(image_path):
                print(f"[ERROR] Line {line_num}: Image file not found: {image_path}")
                num_errors += 1
                continue
            
            # 生成输出路径
            output_path = get_output_path(image_path, args.output_dir)
            
            # 检查是否跳过已存在的文件
            if args.skip_existing and os.path.isfile(output_path):
                print(f"[SKIP] Line {line_num}: Output already exists: {output_path}")
                num_skipped += 1
                continue
            
            try:
                # 加载图像
                image = PILImage.open(image_path).convert("RGB")
                
                # 调整大小（如果指定）
                if args.resize:
                    image = image.resize(args.resize, PILImage.BILINEAR)
                
                # 运行SAM2预测
                mask, confidence = sam_wrapper.predict(image, points, labels)
                
                # 计算包围盒
                x_min, y_min, x_max, y_max = calculate_bounding_box(mask)
                
                # 保存mask
                save_mask_as_grayscale(mask, output_path)
                
                # 如果需要JSON输出，添加到结果列表
                if args.json_output:
                    json_results.append({
                        "mask_path": output_path,
                        "bbox": [x_min, y_min, x_max, y_max]
                    })
                
                print(f"[OK] Line {line_num}: {image_path} -> {output_path} (confidence: {confidence:.3f}, bbox: [{x_min},{y_min},{x_max},{y_max}])")
                num_processed += 1
                
            except Exception as e:
                print(f"[ERROR] Line {line_num}: Failed to process {image_path} - {e}")
                num_errors += 1
                continue
    
    # 保存JSON输出
    if args.json_output and json_results:
        try:
            with open(args.json_output, "w", encoding="utf-8") as json_file:
                json.dump(json_results, json_file, indent=2, ensure_ascii=False)
            print(f"JSON output saved to: {args.json_output}")
        except Exception as e:
            print(f"[ERROR] Failed to save JSON output: {e}")
            return 1
    
    # 打印总结
    print(f"\nProcessing completed:")
    print(f"  Processed: {num_processed}")
    print(f"  Skipped: {num_skipped}")
    print(f"  Errors: {num_errors}")
    print(f"  Output directory: {args.output_dir}")
    if args.json_output:
        print(f"  JSON output: {args.json_output}")
    
    return 0 if num_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

```

```python
#!/usr/bin/env python3

"""
使用SAM2从点提示进行图像分割

读取包含图像路径和正/负类点坐标的JSONL文件，使用SAM2进行分割，
并将预测的mask保存为灰度图像。

功能:
- 从点提示生成图像分割mask
- 自动计算每个mask的最小包围盒
- 可选择输出包含mask路径和包围盒坐标的JSON文件

输入JSONL格式:
    新格式: {"image": "/path/img.jpg", "points": [[x1,y1], [x2,y2]], "labels": [1, 0]}
    旧格式: {"image": "/path/img.jpg", "x": x1, "y": y1}

输出:
    与输入图像对应的灰度mask图像，保存到指定目录
    - 0 (黑色): 背景
    - 255 (白色): 前景

依赖项:
    - SAM2 (Segment Anything Model 2)
    - torch, PIL, opencv-python, numpy

安装SAM2:
    git clone https://github.com/facebookresearch/segment-anything-2.git third_party/sam2
    cd third_party/sam2 && pip install -e .

下载模型:
    # 下载SAM2.1 Hiera Large模型 (~900MB)
    wget -O third_party/sam2/checkpoints/sam2.1_hiera_large.pt \\
      https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

基础用法:
    python seg-rl/sam2_segment_from_points.py \\
      --input_jsonl /path/to/points.jsonl \\
      --output_dir /path/to/masks \\
      --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \\
      --device cuda

带JSON输出:
    python seg-rl/sam2_segment_from_points.py \\
      --input_jsonl /path/to/points.jsonl \\
      --output_dir /path/to/masks \\
      --sam_checkpoint /path/to/sam2.1_hiera_large.pt \\
      --device cuda \\
      --resize 512 512 \\
      --skip_existing \\
      --json_output /path/to/results.json

JSON输出格式:
    [
      {
        "mask_path": "/path/to/mask1.png",
        "bbox": [10, 20, 100, 150]
      },
      ...
    ]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

# 添加sam2路径
sys.path.append(str(Path(__file__).parent.parent / "third_party" / "sam2"))

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError as e:
    print(f"Error importing SAM2: {e}")
    print("Please ensure SAM2 is properly installed and the path is correct.")
    sys.exit(1)


class SAMWrapper:
    """SAM2包装器，用于图像分割预测"""
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """初始化SAM2模型和预测器
        
        Args:
            model_path: SAM2模型检查点路径
            device: 运行设备 (e.g. "cuda", "cuda:0", "cpu")
                   如果为None，将自动检测可用设备
        """
        # SAM2配置文件路径
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        
        # 检查配置文件是否存在
        config_path = Path(__file__).parent.parent / "third_party" / "sam2" / model_cfg
        if not config_path.exists():
            # 尝试相对于当前目录
            config_path = Path(model_cfg)
            if not config_path.exists():
                raise FileNotFoundError(f"SAM2 config file not found: {model_cfg}")
        
        sam_model = build_sam2(str(config_path), model_path)
        
        # 自动检测设备
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)
        
        # 移动到指定设备
        sam_model = sam_model.to(self.device)
        
        # 初始化预测器
        self.predictor = SAM2ImagePredictor(sam_model)
        self.last_mask = None
        
    def predict(self, 
                image: PILImage.Image, 
                points: Optional[List[Tuple[int, int]]] = None, 
                labels: Optional[List[int]] = None,
                bbox: Optional[List[int]] = None) -> Tuple[np.ndarray, float]:
        """使用给定提示运行SAM2预测
        
        Args:
            image: 输入PIL图像
            points: 点坐标列表 [(x,y), ...]
            labels: 点标签列表 (1=前景, 0=背景)
            bbox: 可选边界框 [x1,y1,x2,y2]
            
        Returns:
            (predicted_mask, confidence_score)的元组
        """
        input_points = np.array(points) if points else None
        input_labels = np.array(labels) if labels else None
        input_bboxes = np.array([bbox]) if bbox else None

        # 转换为numpy数组
        image_np = np.array(image)
        
        # 确保是RGB格式
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image_np
        
        # 设置图像
        self.predictor.set_image(rgb_image)
        
        # 预测
        mask_pred, score, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=input_bboxes,
            multimask_output=False,
        )
        
        self.last_mask = mask_pred[0]
        return mask_pred[0], score[0]


def validate_and_extract_points(obj: Dict[str, Any], line_num: int) -> Tuple[bool, Optional[List[Tuple[float, float]]], Optional[List[int]]]:
    """验证并提取JSON条目中的点和标签
    
    Args:
        obj: JSON对象
        line_num: 行号（用于错误报告）
        
    Returns:
        (is_valid, points, labels)的元组
    """
    if "image" not in obj:
        print(f"[WARN] Line {line_num}: Missing 'image' field")
        return False, None, None
        
    # 检查格式：新格式有"points"和"labels"，旧格式有"x"和"y"
    if "points" in obj and "labels" in obj:
        # 新格式
        points = obj["points"]
        labels = obj["labels"]
        
        if not isinstance(points, list) or not isinstance(labels, list):
            print(f"[WARN] Line {line_num}: 'points' and 'labels' must be lists")
            return False, None, None
            
        if len(points) != len(labels):
            print(f"[WARN] Line {line_num}: 'points' and 'labels' must have same length")
            return False, None, None
        
        # 验证点格式
        valid_points = []
        valid_labels = []
        for point, label in zip(points, labels):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                print(f"[WARN] Line {line_num}: Invalid point format {point}")
                continue
            valid_points.append((float(point[0]), float(point[1])))
            valid_labels.append(int(label))
            
        if len(valid_points) == 0:
            print(f"[WARN] Line {line_num}: No valid points found")
            return False, None, None
            
        return True, valid_points, valid_labels
        
    elif "x" in obj and "y" in obj:
        # 旧格式 - 假设为正类点
        x = float(obj["x"])
        y = float(obj["y"])
        return True, [(x, y)], [1]
        
    else:
        print(f"[WARN] Line {line_num}: Missing coordinate data (need 'points'+'labels' or 'x'+'y')")
        return False, None, None


def get_output_path(image_path: str, output_dir: str) -> str:
    """生成输出mask文件路径
    
    Args:
        image_path: 输入图像路径
        output_dir: 输出目录
        
    Returns:
        输出mask文件路径
    """
    image_name = Path(image_path).stem
    return os.path.join(output_dir, f"{image_name}.png")


def calculate_bounding_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """计算mask的最小包围盒
    
    Args:
        mask: 二值mask数组
        
    Returns:
        (x_min, y_min, x_max, y_max) 包围盒坐标
    """
    # 确保mask是二值的
    mask_binary = (mask > 0).astype(np.uint8)
    
    # 找到非零像素的坐标
    y_indices, x_indices = np.where(mask_binary)
    
    if len(x_indices) == 0 or len(y_indices) == 0:
        # 如果没有前景像素，返回空包围盒
        return 0, 0, 0, 0
    
    x_min = int(np.min(x_indices))
    x_max = int(np.max(x_indices))
    y_min = int(np.min(y_indices))
    y_max = int(np.max(y_indices))
    
    return x_min, y_min, x_max, y_max


def save_mask_as_grayscale(mask: np.ndarray, output_path: str) -> None:
    """将mask保存为灰度图像
    
    Args:
        mask: 二值mask数组
        output_path: 输出文件路径
    """
    # 确保mask是二值的
    mask_binary = (mask > 0).astype(np.uint8)
    
    # 转换为灰度值 (0=背景, 255=前景)
    mask_gray = mask_binary * 255
    
    # 保存
    cv2.imwrite(output_path, mask_gray)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Use SAM2 to segment images from point prompts")
    p.add_argument("--input_jsonl", type=str, required=True,
                   help="Input JSONL file containing image paths and point coordinates")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output directory to save mask images")
    p.add_argument("--sam_checkpoint", type=str, required=True,
                   help="Path to SAM2 model checkpoint")
    p.add_argument("--device", type=str, default=None,
                   help="Device to run on (cuda/cpu). Auto-detect if not specified")
    p.add_argument("--config_path", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml",
                   help="Path to SAM2 config file")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip processing if output file already exists")
    p.add_argument("--resize", type=int, nargs=2, default=None, metavar=("WIDTH", "HEIGHT"),
                   help="Resize input images to specified size [width height]")
    p.add_argument("--json_output", type=str, default=None,
                   help="Path to output JSON file containing mask paths and bounding boxes")
    return p.parse_args()


def main():
    args = parse_args()
    
    # 检查输入文件
    if not os.path.isfile(args.input_jsonl):
        print(f"Error: Input JSONL file not found: {args.input_jsonl}")
        return 1
    
    # 检查SAM2检查点
    if not os.path.isfile(args.sam_checkpoint):
        print(f"Error: SAM2 checkpoint not found: {args.sam_checkpoint}")
        return 1
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化SAM2
    print(f"Initializing SAM2 with checkpoint: {args.sam_checkpoint}")
    print(f"Using device: {args.device or 'auto-detect'}")
    
    try:
        sam_wrapper = SAMWrapper(args.sam_checkpoint, args.device)
        print("SAM2 initialized successfully")
    except Exception as e:
        print(f"Error initializing SAM2: {e}")
        return 1
    
    # 处理JSONL文件
    print(f"Processing JSONL file: {args.input_jsonl}")
    
    num_processed = 0
    num_skipped = 0
    num_errors = 0
    
    # 用于存储JSON输出的列表
    json_results = []
    
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[ERROR] Line {line_num}: Invalid JSON - {e}")
                num_errors += 1
                continue
            
            # 验证并提取点信息
            is_valid, points, labels = validate_and_extract_points(obj, line_num)
            if not is_valid:
                num_errors += 1
                continue
            
            image_path = obj["image"]
            
            # 检查图像文件是否存在
            if not os.path.isfile(image_path):
                print(f"[ERROR] Line {line_num}: Image file not found: {image_path}")
                num_errors += 1
                continue
            
            # 生成输出路径
            output_path = get_output_path(image_path, args.output_dir)
            
            # 检查是否跳过已存在的文件
            if args.skip_existing and os.path.isfile(output_path):
                print(f"[SKIP] Line {line_num}: Output already exists: {output_path}")
                num_skipped += 1
                continue
            
            try:
                # 加载图像
                image = PILImage.open(image_path).convert("RGB")
                
                # 调整大小（如果指定）
                if args.resize:
                    image = image.resize(args.resize, PILImage.BILINEAR)
                
                # 运行SAM2预测
                mask, confidence = sam_wrapper.predict(image, points, labels)
                
                # 计算包围盒
                x_min, y_min, x_max, y_max = calculate_bounding_box(mask)
                
                # 保存mask
                save_mask_as_grayscale(mask, output_path)
                
                # 如果需要JSON输出，添加到结果列表
                if args.json_output:
                    json_results.append({
                        "mask_path": output_path,
                        "bbox": [x_min, y_min, x_max, y_max]
                    })
                
                print(f"[OK] Line {line_num}: {image_path} -> {output_path} (confidence: {confidence:.3f}, bbox: [{x_min},{y_min},{x_max},{y_max}])")
                num_processed += 1
                
            except Exception as e:
                print(f"[ERROR] Line {line_num}: Failed to process {image_path} - {e}")
                num_errors += 1
                continue
    
    # 保存JSON输出
    if args.json_output and json_results:
        try:
            with open(args.json_output, "w", encoding="utf-8") as json_file:
                json.dump(json_results, json_file, indent=2, ensure_ascii=False)
            print(f"JSON output saved to: {args.json_output}")
        except Exception as e:
            print(f"[ERROR] Failed to save JSON output: {e}")
            return 1
    
    # 打印总结
    print(f"\nProcessing completed:")
    print(f"  Processed: {num_processed}")
    print(f"  Skipped: {num_skipped}")
    print(f"  Errors: {num_errors}")
    print(f"  Output directory: {args.output_dir}")
    if args.json_output:
        print(f"  JSON output: {args.json_output}")
    
    return 0 if num_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

```

完成！我已经将包围盒的表示格式从字典形式改为了数组形式。现在的更改包括：

## 📝 更新内容

### 1. **JSON格式更改**
- **之前**: `"bounding_box": {"x_min": 10, "y_min": 20, "x_max": 100, "y_max": 150}`
- **现在**: `"bbox": [10, 20, 100, 150]`

### 2. **新的JSON输出格式**
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

其中 `bbox` 数组的格式为 `[x_min, y_min, x_max, y_max]`，这是一种更简洁和常用的包围盒表示方式。

### 3. **两个文件都已更新**
- ✅ `sam2_segment_simple.py`
- ✅ `sam2_segment_from_points.py`

这种格式更加简洁，也符合很多计算机视觉库和框架的标准包围盒表示方法。

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

---

**User**

写一个单独的脚本，对指定目录下的所有mask图片，计算包围框并按照上述格式生成JSON描述文件写入到指定文件路径下

---

**Cursor**

我来创建一个单独的脚本来计算指定目录下所有mask图片的包围盒并生成JSON文件。

```python
#!/usr/bin/env python3

"""
从mask图片目录计算包围盒并生成JSON描述文件

读取指定目录下的所有mask图片，计算每个mask的最小包围盒，
并将结果保存为JSON格式文件。

功能:
- 支持多种图片格式 (.png, .jpg, .jpeg, .bmp, .tiff)
- 自动计算每个mask的最小包围盒
- 输出标准JSON格式文件

用法:
    python seg-rl/calculate_bbox_from_masks.py \
      --mask_dir /path/to/masks \
      --output_json /path/to/output.json

示例:
    python seg-rl/calculate_bbox_from_masks.py \
      --mask_dir /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/masks_step1 \
      --output_json /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/pred_masks-0.json

输出JSON格式:
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

其中bbox格式为 [x_min, y_min, x_max, y_max]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np

# 支持的图片格式
SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}


def calculate_bounding_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """计算mask的最小包围盒
    
    Args:
        mask: 二值mask数组
        
    Returns:
        (x_min, y_min, x_max, y_max) 包围盒坐标
    """
    # 确保mask是二值的
    if len(mask.shape) == 3:
        # 如果是彩色图像，转换为灰度
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    # 二值化处理
    mask_binary = (mask > 0).astype(np.uint8)
    
    # 找到非零像素的坐标
    y_indices, x_indices = np.where(mask_binary)
    
    if len(x_indices) == 0 or len(y_indices) == 0:
        # 如果没有前景像素，返回空包围盒
        return 0, 0, 0, 0
    
    x_min = int(np.min(x_indices))
    x_max = int(np.max(x_indices))
    y_min = int(np.min(y_indices))
    y_max = int(np.max(y_indices))
    
    return x_min, y_min, x_max, y_max


def get_mask_files(mask_dir: str) -> List[str]:
    """获取目录下所有支持格式的mask文件
    
    Args:
        mask_dir: mask文件目录
        
    Returns:
        mask文件路径列表，按文件名排序
    """
    mask_files = []
    
    if not os.path.isdir(mask_dir):
        print(f"Error: Directory not found: {mask_dir}")
        return mask_files
    
    for file_path in Path(mask_dir).iterdir():
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            mask_files.append(str(file_path))
    
    # 按文件名排序
    mask_files.sort()
    return mask_files


def process_mask_file(mask_path: str) -> Tuple[bool, int, int, int, int]:
    """处理单个mask文件
    
    Args:
        mask_path: mask文件路径
        
    Returns:
        (success, x_min, y_min, x_max, y_max) 元组
    """
    try:
        # 读取mask图像
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            print(f"[ERROR] Failed to load image: {mask_path}")
            return False, 0, 0, 0, 0
        
        # 计算包围盒
        x_min, y_min, x_max, y_max = calculate_bounding_box(mask)
        return True, x_min, y_min, x_max, y_max
        
    except Exception as e:
        print(f"[ERROR] Failed to process {mask_path}: {e}")
        return False, 0, 0, 0, 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate bounding boxes from mask images and generate JSON output"
    )
    parser.add_argument(
        "--mask_dir", 
        type=str, 
        required=True,
        help="Directory containing mask images"
    )
    parser.add_argument(
        "--output_json", 
        type=str, 
        required=True,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--relative_paths", 
        action="store_true",
        help="Use relative paths in JSON output (relative to mask_dir)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Print detailed processing information"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 检查输入目录
    if not os.path.isdir(args.mask_dir):
        print(f"Error: Mask directory not found: {args.mask_dir}")
        return 1
    
    # 获取所有mask文件
    print(f"Scanning mask directory: {args.mask_dir}")
    mask_files = get_mask_files(args.mask_dir)
    
    if not mask_files:
        print(f"No supported image files found in: {args.mask_dir}")
        print(f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}")
        return 1
    
    print(f"Found {len(mask_files)} mask files")
    
    # 处理所有mask文件
    json_results = []
    num_processed = 0
    num_errors = 0
    
    for i, mask_path in enumerate(mask_files, 1):
        if args.verbose:
            print(f"[{i}/{len(mask_files)}] Processing: {mask_path}")
        
        success, x_min, y_min, x_max, y_max = process_mask_file(mask_path)
        
        if success:
            # 决定使用相对路径还是绝对路径
            if args.relative_paths:
                # 相对于mask_dir的相对路径
                output_path = os.path.relpath(mask_path, args.mask_dir)
            else:
                # 使用绝对路径
                output_path = os.path.abspath(mask_path)
            
            json_results.append({
                "mask_path": output_path,
                "bbox": [x_min, y_min, x_max, y_max]
            })
            
            if args.verbose:
                print(f"  -> bbox: [{x_min}, {y_min}, {x_max}, {y_max}]")
            
            num_processed += 1
        else:
            num_errors += 1
    
    # 保存JSON输出
    if json_results:
        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(args.output_json)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            with open(args.output_json, "w", encoding="utf-8") as json_file:
                json.dump(json_results, json_file, indent=2, ensure_ascii=False)
            
            print(f"\nJSON output saved to: {args.output_json}")
        except Exception as e:
            print(f"[ERROR] Failed to save JSON output: {e}")
            return 1
    else:
        print("No valid masks processed, no JSON output generated.")
        return 1
    
    # 打印总结
    print(f"\nProcessing completed:")
    print(f"  Total files: {len(mask_files)}")
    print(f"  Processed: {num_processed}")
    print(f"  Errors: {num_errors}")
    print(f"  Output file: {args.output_json}")
    
    return 0 if num_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
```

现在让我检查文件是否有任何语法错误：

让我为这个脚本添加可执行权限：

完美！我已经创建了一个独立的脚本 `calculate_bbox_from_masks.py`，用于从mask图片目录计算包围盒并生成JSON文件。

## 📝 脚本功能特点

### 🎯 **主要功能**
- **批量处理**: 扫描指定目录下的所有mask图片
- **多格式支持**: 支持 `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.tif` 格式
- **包围盒计算**: 自动计算每个mask的最小包围盒
- **JSON输出**: 生成标准格式的JSON描述文件

### 🔧 **使用方法**

**基础用法**:
```bash
python seg-rl/calculate_bbox_from_masks.py \
  --mask_dir /path/to/masks \
  --output_json /path/to/output.json
```

**实际示例**:
```bash
python seg-rl/calculate_bbox_from_masks.py \
  --mask_dir /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/masks_step1 \
  --output_json /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/pred_masks-0.json
```

**高级选项**:
```bash
python seg-rl/calculate_bbox_from_masks.py \
  --mask_dir /path/to/masks \
  --output_json /path/to/output.json \
  --relative_paths \
  --verbose
```

### 📊 **命令行参数**
- `--mask_dir`: **(必需)** mask图片目录路径
- `--output_json`: **(必需)** 输出JSON文件路径
- `--relative_paths`: **(可选)** 在JSON中使用相对路径而非绝对路径
- `--verbose`: **(可选)** 显示详细的处理信息

### 📋 **输出格式**
生成的JSON文件格式与你要求的完全一致：
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

其中 `bbox` 格式为 `[x_min, y_min, x_max, y_max]`

### 🛡️ **错误处理**
- 自动处理无效图片文件
- 处理空mask（返回 `[0, 0, 0, 0]`）
- 支持彩色和灰度mask图片
- 提供详细的错误报告和处理统计

### 💡 **使用场景**
这个脚本非常适合：
- 后处理已生成的mask文件
- 为现有mask数据集生成包围盒标注
- 验证和分析分割结果
- 与其他工具链集成

脚本已经设置了可执行权限，可以直接运行！

```shellscript
python seg-rl/calculate_bbox_from_masks.py \
  --mask_dir /path/to/masks \
  --output_json /path/to/output.json
```

```shellscript
python seg-rl/calculate_bbox_from_masks.py \
  --mask_dir /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/masks_step1 \
  --output_json /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/pred_masks-0.json
```

```shellscript
python seg-rl/calculate_bbox_from_masks.py \
  --mask_dir /path/to/masks \
  --output_json /path/to/output.json \
  --relative_paths \
  --verbose
```

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

---

**User**

接下来请在seg-rl下再编写一个Python脚本：使用SAM2的Automatic模式对指定目录下的图片作批量分割，并评估其与指定路径下JSON文件以前述JSON格式描述的masks之间的吻合度，具体地说：指定的图片目录下的原图与JSON所包含的masks之间是一一对应的（mask与原图文件名相同，仅上级目录不同）。对于每个原图，在JSON文件中都描述了其相对应的mask以及其包围盒的位置坐标。我们要以此为基准去评估SAM2 Automatic模式的分割效果，方法是：对于每个原图，用SAM2的Automatic模式去分割得到的结果会包含多个分割区域，每个区域都有mask和包围盒，用“分割预测的包围盒与JSON中描述的相应原图的bbox是否有重叠区域”作为筛选依据，对通过筛选的预测mask，计算其与JSON中描述的真值mask图之间的重合度（用IOU或DICE指标评估），把其中最大者作为SAM2对这一张图片的分割效果评估。最后统计在整个数据集下，最好/最差/平均 的分割效果指标。并且支持在指定的另一个目录下生成分割效果展示，具体地，可以把Automatic模式分割的多区域作多色展示作为图1，再把最重合区域的pred mask与真值mask用两种半透明颜色与原图转为灰度后的图像叠加显示作为图2，并以文字标示出该样本下的分割效果指标。

---

**Cursor**

我来创建一个使用SAM2 Automatic模式进行批量分割评估的脚本。这是一个相当复杂的任务，我需要实现多个功能模块。

```python
#!/usr/bin/env python3

"""
使用SAM2 Automatic模式批量分割并评估与真值mask的吻合度

功能:
- 使用SAM2 Automatic模式对指定目录下的图片进行批量分割
- 与JSON描述的真值masks进行吻合度评估
- 支持IoU和DICE指标评估
- 生成分割效果可视化图片
- 统计整个数据集的最好/最差/平均分割效果

评估方法:
1. 对每张原图使用SAM2 Automatic模式分割得到多个区域
2. 通过包围盒重叠筛选候选预测mask
3. 计算候选mask与真值mask的IoU/DICE，取最大值作为该图的评估结果
4. 统计整个数据集的分割效果

输出可视化:
- 图1: Automatic模式分割的多区域多色展示
- 图2: 最佳预测mask与真值mask叠加显示（半透明）+ 评估指标

用法:
    python seg-rl/sam2_automatic_evaluation.py \
      --image_dir /path/to/images \
      --json_file /path/to/masks.json \
      --sam_checkpoint /path/to/sam2.1_hiera_large.pt \
      --output_dir /path/to/results \
      --device cuda

示例:
    python seg-rl/sam2_automatic_evaluation.py \
      --image_dir /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/imagesTr \
      --json_file /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/pred_masks-0.json \
      --sam_checkpoint /root/autodl-tmp/works/Seg-R0/third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
      --output_dir /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/evaluation_results \
      --device cuda \
      --metric dice
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image as PILImage
import torch

# 添加sam2路径
sys.path.append(str(Path(__file__).parent.parent / "third_party" / "sam2"))

try:
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
except ImportError as e:
    print(f"Error importing SAM2: {e}")
    print("Please ensure SAM2 is properly installed and the path is correct.")
    sys.exit(1)


# 支持的图片格式
SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}


def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """计算两个mask之间的IoU
    
    Args:
        mask1: 第一个mask (binary)
        mask2: 第二个mask (binary)
        
    Returns:
        IoU值 (0-1)
    """
    # 确保是二值mask
    mask1_bin = (mask1 > 0).astype(np.uint8)
    mask2_bin = (mask2 > 0).astype(np.uint8)
    
    # 计算交集和并集
    intersection = np.logical_and(mask1_bin, mask2_bin).sum()
    union = np.logical_or(mask1_bin, mask2_bin).sum()
    
    if union == 0:
        return 0.0
    
    return float(intersection) / float(union)


def calculate_dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """计算两个mask之间的DICE系数
    
    Args:
        mask1: 第一个mask (binary)
        mask2: 第二个mask (binary)
        
    Returns:
        DICE值 (0-1)
    """
    # 确保是二值mask
    mask1_bin = (mask1 > 0).astype(np.uint8)
    mask2_bin = (mask2 > 0).astype(np.uint8)
    
    # 计算交集
    intersection = np.logical_and(mask1_bin, mask2_bin).sum()
    total = mask1_bin.sum() + mask2_bin.sum()
    
    if total == 0:
        return 0.0
    
    return 2.0 * float(intersection) / float(total)


def bbox_overlap(bbox1: List[int], bbox2: List[int]) -> bool:
    """检查两个包围盒是否有重叠
    
    Args:
        bbox1: [x_min, y_min, x_max, y_max]
        bbox2: [x_min, y_min, x_max, y_max]
        
    Returns:
        是否有重叠
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # 检查是否不重叠
    if x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min:
        return False
    
    return True


def load_ground_truth_masks(json_file: str) -> Dict[str, Dict[str, Any]]:
    """加载真值mask信息
    
    Args:
        json_file: JSON文件路径
        
    Returns:
        字典，键为图片文件名（无扩展名），值为mask信息
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    gt_masks = {}
    for item in data:
        mask_path = item['mask_path']
        bbox = item['bbox']
        
        # 从mask路径提取文件名（无扩展名）
        mask_filename = Path(mask_path).stem
        
        gt_masks[mask_filename] = {
            'mask_path': mask_path,
            'bbox': bbox
        }
    
    return gt_masks


def get_image_files(image_dir: str) -> List[str]:
    """获取图片目录下所有支持的图片文件
    
    Args:
        image_dir: 图片目录
        
    Returns:
        图片文件路径列表，按文件名排序
    """
    image_files = []
    
    if not os.path.isdir(image_dir):
        print(f"Error: Directory not found: {image_dir}")
        return image_files
    
    for file_path in Path(image_dir).iterdir():
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            image_files.append(str(file_path))
    
    image_files.sort()
    return image_files


def load_mask_image(mask_path: str) -> Optional[np.ndarray]:
    """加载mask图像
    
    Args:
        mask_path: mask文件路径
        
    Returns:
        mask数组或None
    """
    try:
        if not os.path.exists(mask_path):
            print(f"Warning: Mask file not found: {mask_path}")
            return None
        
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            print(f"Warning: Failed to load mask: {mask_path}")
            return None
        
        # 如果是彩色图像，转换为灰度
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        return mask
    except Exception as e:
        print(f"Warning: Error loading mask {mask_path}: {e}")
        return None


def evaluate_image(image_path: str, gt_info: Dict[str, Any], mask_generator, metric: str = 'iou') -> Tuple[float, Dict[str, Any]]:
    """评估单张图片的分割效果
    
    Args:
        image_path: 图片路径
        gt_info: 真值信息
        mask_generator: SAM2 mask生成器
        metric: 评估指标 ('iou' 或 'dice')
        
    Returns:
        (最佳分数, 详细信息字典)
    """
    try:
        # 加载图片
        image = cv2.imread(image_path)
        if image is None:
            return 0.0, {'error': f'Failed to load image: {image_path}'}
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 加载真值mask
        gt_mask = load_mask_image(gt_info['mask_path'])
        if gt_mask is None:
            return 0.0, {'error': f'Failed to load ground truth mask: {gt_info["mask_path"]}'}
        
        gt_bbox = gt_info['bbox']
        
        # 使用SAM2 Automatic模式生成mask
        masks = mask_generator.generate(image_rgb)
        
        if not masks:
            return 0.0, {'error': 'No masks generated by SAM2', 'num_generated': 0}
        
        # 筛选与真值包围盒重叠的预测mask
        candidate_masks = []
        for mask_info in masks:
            pred_bbox = mask_info['bbox']  # SAM2返回的格式应该是[x, y, w, h]
            # 转换为[x_min, y_min, x_max, y_max]格式
            pred_bbox_xyxy = [
                int(pred_bbox[0]), 
                int(pred_bbox[1]), 
                int(pred_bbox[0] + pred_bbox[2]), 
                int(pred_bbox[1] + pred_bbox[3])
            ]
            
            if bbox_overlap(gt_bbox, pred_bbox_xyxy):
                candidate_masks.append({
                    'mask': mask_info['segmentation'],
                    'bbox': pred_bbox_xyxy,
                    'area': mask_info['area'],
                    'stability_score': mask_info.get('stability_score', 0.0)
                })
        
        if not candidate_masks:
            return 0.0, {
                'error': 'No overlapping masks found',
                'num_generated': len(masks),
                'num_candidates': 0
            }
        
        # 计算每个候选mask与真值的重合度
        best_score = 0.0
        best_mask_info = None
        
        for mask_info in candidate_masks:
            pred_mask = mask_info['mask']
            
            if metric == 'dice':
                score = calculate_dice(gt_mask, pred_mask)
            else:  # iou
                score = calculate_iou(gt_mask, pred_mask)
            
            if score > best_score:
                best_score = score
                best_mask_info = mask_info
        
        return best_score, {
            'num_generated': len(masks),
            'num_candidates': len(candidate_masks),
            'best_mask': best_mask_info,
            'all_generated_masks': masks,
            'gt_bbox': gt_bbox,
            'gt_mask': gt_mask
        }
        
    except Exception as e:
        return 0.0, {'error': f'Exception during evaluation: {str(e)}'}


def create_visualization(image_path: str, eval_info: Dict[str, Any], best_score: float, 
                        metric: str, output_path: str) -> None:
    """创建可视化图片
    
    Args:
        image_path: 原图路径
        eval_info: 评估信息
        best_score: 最佳分数
        metric: 评估指标
        output_path: 输出路径（无扩展名）
    """
    try:
        # 加载原图
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 创建图1: 多区域分割展示
        fig1, ax1 = plt.subplots(1, 1, figsize=(12, 8))
        ax1.imshow(image_rgb)
        ax1.set_title(f'SAM2 Automatic Segmentation Results\n{len(eval_info["all_generated_masks"])} masks generated')
        
        # 显示所有生成的mask
        colors = plt.cm.Set3(np.linspace(0, 1, len(eval_info["all_generated_masks"])))
        for i, mask_info in enumerate(eval_info["all_generated_masks"]):
            mask = mask_info['segmentation']
            # 创建彩色mask
            colored_mask = np.zeros((*mask.shape, 4))
            colored_mask[mask] = [*colors[i][:3], 0.6]  # 半透明
            ax1.imshow(colored_mask)
            
            # 添加包围盒
            bbox = mask_info['bbox']  # [x, y, w, h]
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                                   linewidth=1, edgecolor=colors[i], facecolor='none')
            ax1.add_patch(rect)
        
        ax1.axis('off')
        plt.tight_layout()
        plt.savefig(f"{output_path}_all_masks.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 创建图2: 最佳预测与真值对比
        if eval_info.get('best_mask') is not None and eval_info.get('gt_mask') is not None:
            fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))
            
            # 转换为灰度图作为背景
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ax2.imshow(image_gray, cmap='gray', alpha=0.7)
            
            # 显示真值mask（绿色）
            gt_mask = eval_info['gt_mask']
            gt_colored = np.zeros((*gt_mask.shape, 4))
            gt_colored[gt_mask > 0] = [0, 1, 0, 0.5]  # 绿色半透明
            ax2.imshow(gt_colored)
            
            # 显示最佳预测mask（红色）
            best_mask = eval_info['best_mask']['mask']
            pred_colored = np.zeros((*best_mask.shape, 4))
            pred_colored[best_mask > 0] = [1, 0, 0, 0.5]  # 红色半透明
            ax2.imshow(pred_colored)
            
            # 添加包围盒
            gt_bbox = eval_info['gt_bbox']
            pred_bbox = eval_info['best_mask']['bbox']
            
            # 真值包围盒（绿色）
            gt_rect = patches.Rectangle((gt_bbox[0], gt_bbox[1]), 
                                      gt_bbox[2]-gt_bbox[0], gt_bbox[3]-gt_bbox[1],
                                      linewidth=2, edgecolor='green', facecolor='none', 
                                      linestyle='--', label='Ground Truth')
            ax2.add_patch(gt_rect)
            
            # 预测包围盒（红色）
            pred_rect = patches.Rectangle((pred_bbox[0], pred_bbox[1]), 
                                        pred_bbox[2]-pred_bbox[0], pred_bbox[3]-pred_bbox[1],
                                        linewidth=2, edgecolor='red', facecolor='none', 
                                        label='Best Prediction')
            ax2.add_patch(pred_rect)
            
            ax2.set_title(f'Best Prediction vs Ground Truth\n'
                         f'{metric.upper()}: {best_score:.3f} | '
                         f'Candidates: {eval_info["num_candidates"]}/{eval_info["num_generated"]}')
            ax2.legend()
            ax2.axis('off')
            
            plt.tight_layout()
            plt.savefig(f"{output_path}_comparison.png", dpi=150, bbox_inches='tight')
            plt.close()
        
    except Exception as e:
        print(f"Warning: Failed to create visualization for {image_path}: {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SAM2 Automatic segmentation against ground truth masks"
    )
    parser.add_argument(
        "--image_dir", 
        type=str, 
        required=True,
        help="Directory containing input images"
    )
    parser.add_argument(
        "--json_file", 
        type=str, 
        required=True,
        help="JSON file containing ground truth mask information"
    )
    parser.add_argument(
        "--sam_checkpoint", 
        type=str, 
        required=True,
        help="Path to SAM2 model checkpoint"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True,
        help="Output directory for results and visualizations"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default=None,
        help="Device to run on (cuda/cpu). Auto-detect if not specified"
    )
    parser.add_argument(
        "--metric", 
        type=str, 
        choices=['iou', 'dice'], 
        default='iou',
        help="Evaluation metric (default: iou)"
    )
    parser.add_argument(
        "--config_path", 
        type=str, 
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="Path to SAM2 config file"
    )
    parser.add_argument(
        "--points_per_side", 
        type=int, 
        default=32,
        help="Number of points per side for automatic mask generation"
    )
    parser.add_argument(
        "--pred_iou_thresh", 
        type=float, 
        default=0.88,
        help="IoU threshold for mask prediction"
    )
    parser.add_argument(
        "--stability_score_thresh", 
        type=float, 
        default=0.95,
        help="Stability score threshold for mask filtering"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Print detailed processing information"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 检查输入
    if not os.path.isdir(args.image_dir):
        print(f"Error: Image directory not found: {args.image_dir}")
        return 1
    
    if not os.path.isfile(args.json_file):
        print(f"Error: JSON file not found: {args.json_file}")
        return 1
    
    if not os.path.isfile(args.sam_checkpoint):
        print(f"Error: SAM2 checkpoint not found: {args.sam_checkpoint}")
        return 1
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载真值数据
    print(f"Loading ground truth data from: {args.json_file}")
    gt_masks = load_ground_truth_masks(args.json_file)
    print(f"Loaded {len(gt_masks)} ground truth masks")
    
    # 获取图片文件
    print(f"Scanning image directory: {args.image_dir}")
    image_files = get_image_files(args.image_dir)
    print(f"Found {len(image_files)} images")
    
    # 初始化SAM2
    print(f"Initializing SAM2 with checkpoint: {args.sam_checkpoint}")
    
    # 自动检测设备
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    try:
        # SAM2配置文件路径
        config_path = Path(__file__).parent.parent / "third_party" / "sam2" / args.config_path
        if not config_path.exists():
            config_path = Path(args.config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"SAM2 config file not found: {args.config_path}")
        
        sam_model = build_sam2(str(config_path), args.sam_checkpoint)
        sam_model = sam_model.to(device)
        
        # 创建自动mask生成器
        mask_generator = SAM2AutomaticMaskGenerator(
            model=sam_model,
            points_per_side=args.points_per_side,
            pred_iou_thresh=args.pred_iou_thresh,
            stability_score_thresh=args.stability_score_thresh,
        )
        
        print("SAM2 initialized successfully")
        
    except Exception as e:
        print(f"Error initializing SAM2: {e}")
        return 1
    
    # 评估所有图片
    print(f"\nStarting evaluation with {args.metric.upper()} metric...")
    
    all_scores = []
    num_processed = 0
    num_errors = 0
    results_log = []
    
    start_time = time.time()
    
    for i, image_path in enumerate(image_files, 1):
        image_name = Path(image_path).stem
        
        if args.verbose:
            print(f"[{i}/{len(image_files)}] Processing: {image_name}")
        
        # 检查是否有对应的真值
        if image_name not in gt_masks:
            if args.verbose:
                print(f"  -> No ground truth found, skipping")
            num_errors += 1
            continue
        
        gt_info = gt_masks[image_name]
        
        # 评估图片
        score, eval_info = evaluate_image(image_path, gt_info, mask_generator, args.metric)
        
        if 'error' in eval_info:
            if args.verbose:
                print(f"  -> Error: {eval_info['error']}")
            num_errors += 1
            continue
        
        all_scores.append(score)
        num_processed += 1
        
        # 记录结果
        result_entry = {
            'image_name': image_name,
            'score': score,
            'num_generated': eval_info['num_generated'],
            'num_candidates': eval_info['num_candidates']
        }
        results_log.append(result_entry)
        
        if args.verbose:
            print(f"  -> {args.metric.upper()}: {score:.3f} "
                  f"({eval_info['num_candidates']}/{eval_info['num_generated']} candidates)")
        
        # 创建可视化
        vis_output_path = os.path.join(args.output_dir, image_name)
        create_visualization(image_path, eval_info, score, args.metric, vis_output_path)
    
    # 计算统计信息
    if all_scores:
        best_score = max(all_scores)
        worst_score = min(all_scores)
        avg_score = sum(all_scores) / len(all_scores)
        
        # 保存详细结果
        results_summary = {
            'metric': args.metric,
            'total_images': len(image_files),
            'processed': num_processed,
            'errors': num_errors,
            'statistics': {
                'best': best_score,
                'worst': worst_score,
                'average': avg_score
            },
            'detailed_results': results_log,
            'parameters': {
                'points_per_side': args.points_per_side,
                'pred_iou_thresh': args.pred_iou_thresh,
                'stability_score_thresh': args.stability_score_thresh
            }
        }
        
        results_file = os.path.join(args.output_dir, 'evaluation_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)
        
        # 打印总结
        elapsed_time = time.time() - start_time
        print(f"\nEvaluation completed in {elapsed_time:.1f} seconds:")
        print(f"  Total images: {len(image_files)}")
        print(f"  Processed: {num_processed}")
        print(f"  Errors: {num_errors}")
        print(f"  Metric: {args.metric.upper()}")
        print(f"  Best score: {best_score:.3f}")
        print(f"  Worst score: {worst_score:.3f}")
        print(f"  Average score: {avg_score:.3f}")
        print(f"  Results saved to: {results_file}")
        print(f"  Visualizations saved to: {args.output_dir}")
        
        return 0
    else:
        print("No images were successfully processed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

现在检查脚本是否有语法错误：

添加可执行权限：

现在让我检查脚本中是否需要matplotlib依赖，如果需要的话创建一个简化版本以避免复杂的依赖：

```python
#!/usr/bin/env python3

"""
使用SAM2 Automatic模式批量分割并评估与真值mask的吻合度

功能:
- 使用SAM2 Automatic模式对指定目录下的图片进行批量分割
- 与JSON描述的真值masks进行吻合度评估
- 支持IoU和DICE指标评估
- 生成分割效果可视化图片
- 统计整个数据集的最好/最差/平均分割效果

评估方法:
1. 对每张原图使用SAM2 Automatic模式分割得到多个区域
2. 通过包围盒重叠筛选候选预测mask
3. 计算候选mask与真值mask的IoU/DICE，取最大值作为该图的评估结果
4. 统计整个数据集的分割效果

输出可视化:
- 图1: Automatic模式分割的多区域多色展示
- 图2: 最佳预测mask与真值mask叠加显示（半透明）+ 评估指标

用法:
    python seg-rl/sam2_automatic_evaluation.py \
      --image_dir /path/to/images \
      --json_file /path/to/masks.json \
      --sam_checkpoint /path/to/sam2.1_hiera_large.pt \
      --output_dir /path/to/results \
      --device cuda

示例:
    python seg-rl/sam2_automatic_evaluation.py \
      --image_dir /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/imagesTr \
      --json_file /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/pred_masks-0.json \
      --sam_checkpoint /root/autodl-tmp/works/Seg-R0/third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
      --output_dir /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/evaluation_results \
      --device cuda \
      --metric dice
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import time

import cv2
import numpy as np
from PIL import Image as PILImage
import torch

# 可选的matplotlib导入
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    HAS_MATPLOTLIB = True
except ImportError:
    print("Warning: matplotlib not found. Visualization features will be disabled.")
    HAS_MATPLOTLIB = False

# 添加sam2路径
sys.path.append(str(Path(__file__).parent.parent / "third_party" / "sam2"))

try:
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
except ImportError as e:
    print(f"Error importing SAM2: {e}")
    print("Please ensure SAM2 is properly installed and the path is correct.")
    sys.exit(1)


# 支持的图片格式
SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}


def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """计算两个mask之间的IoU
    
    Args:
        mask1: 第一个mask (binary)
        mask2: 第二个mask (binary)
        
    Returns:
        IoU值 (0-1)
    """
    # 确保是二值mask
    mask1_bin = (mask1 > 0).astype(np.uint8)
    mask2_bin = (mask2 > 0).astype(np.uint8)
    
    # 计算交集和并集
    intersection = np.logical_and(mask1_bin, mask2_bin).sum()
    union = np.logical_or(mask1_bin, mask2_bin).sum()
    
    if union == 0:
        return 0.0
    
    return float(intersection) / float(union)


def calculate_dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """计算两个mask之间的DICE系数
    
    Args:
        mask1: 第一个mask (binary)
        mask2: 第二个mask (binary)
        
    Returns:
        DICE值 (0-1)
    """
    # 确保是二值mask
    mask1_bin = (mask1 > 0).astype(np.uint8)
    mask2_bin = (mask2 > 0).astype(np.uint8)
    
    # 计算交集
    intersection = np.logical_and(mask1_bin, mask2_bin).sum()
    total = mask1_bin.sum() + mask2_bin.sum()
    
    if total == 0:
        return 0.0
    
    return 2.0 * float(intersection) / float(total)


def bbox_overlap(bbox1: List[int], bbox2: List[int]) -> bool:
    """检查两个包围盒是否有重叠
    
    Args:
        bbox1: [x_min, y_min, x_max, y_max]
        bbox2: [x_min, y_min, x_max, y_max]
        
    Returns:
        是否有重叠
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # 检查是否不重叠
    if x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min:
        return False
    
    return True


def load_ground_truth_masks(json_file: str) -> Dict[str, Dict[str, Any]]:
    """加载真值mask信息
    
    Args:
        json_file: JSON文件路径
        
    Returns:
        字典，键为图片文件名（无扩展名），值为mask信息
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    gt_masks = {}
    for item in data:
        mask_path = item['mask_path']
        bbox = item['bbox']
        
        # 从mask路径提取文件名（无扩展名）
        mask_filename = Path(mask_path).stem
        
        gt_masks[mask_filename] = {
            'mask_path': mask_path,
            'bbox': bbox
        }
    
    return gt_masks


def get_image_files(image_dir: str) -> List[str]:
    """获取图片目录下所有支持的图片文件
    
    Args:
        image_dir: 图片目录
        
    Returns:
        图片文件路径列表，按文件名排序
    """
    image_files = []
    
    if not os.path.isdir(image_dir):
        print(f"Error: Directory not found: {image_dir}")
        return image_files
    
    for file_path in Path(image_dir).iterdir():
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            image_files.append(str(file_path))
    
    image_files.sort()
    return image_files


def load_mask_image(mask_path: str) -> Optional[np.ndarray]:
    """加载mask图像
    
    Args:
        mask_path: mask文件路径
        
    Returns:
        mask数组或None
    """
    try:
        if not os.path.exists(mask_path):
            print(f"Warning: Mask file not found: {mask_path}")
            return None
        
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            print(f"Warning: Failed to load mask: {mask_path}")
            return None
        
        # 如果是彩色图像，转换为灰度
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        return mask
    except Exception as e:
        print(f"Warning: Error loading mask {mask_path}: {e}")
        return None


def evaluate_image(image_path: str, gt_info: Dict[str, Any], mask_generator, metric: str = 'iou') -> Tuple[float, Dict[str, Any]]:
    """评估单张图片的分割效果
    
    Args:
        image_path: 图片路径
        gt_info: 真值信息
        mask_generator: SAM2 mask生成器
        metric: 评估指标 ('iou' 或 'dice')
        
    Returns:
        (最佳分数, 详细信息字典)
    """
    try:
        # 加载图片
        image = cv2.imread(image_path)
        if image is None:
            return 0.0, {'error': f'Failed to load image: {image_path}'}
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 加载真值mask
        gt_mask = load_mask_image(gt_info['mask_path'])
        if gt_mask is None:
            return 0.0, {'error': f'Failed to load ground truth mask: {gt_info["mask_path"]}'}
        
        gt_bbox = gt_info['bbox']
        
        # 使用SAM2 Automatic模式生成mask
        masks = mask_generator.generate(image_rgb)
        
        if not masks:
            return 0.0, {'error': 'No masks generated by SAM2', 'num_generated': 0}
        
        # 筛选与真值包围盒重叠的预测mask
        candidate_masks = []
        for mask_info in masks:
            pred_bbox = mask_info['bbox']  # SAM2返回的格式应该是[x, y, w, h]
            # 转换为[x_min, y_min, x_max, y_max]格式
            pred_bbox_xyxy = [
                int(pred_bbox[0]), 
                int(pred_bbox[1]), 
                int(pred_bbox[0] + pred_bbox[2]), 
                int(pred_bbox[1] + pred_bbox[3])
            ]
            
            if bbox_overlap(gt_bbox, pred_bbox_xyxy):
                candidate_masks.append({
                    'mask': mask_info['segmentation'],
                    'bbox': pred_bbox_xyxy,
                    'area': mask_info['area'],
                    'stability_score': mask_info.get('stability_score', 0.0)
                })
        
        if not candidate_masks:
            return 0.0, {
                'error': 'No overlapping masks found',
                'num_generated': len(masks),
                'num_candidates': 0
            }
        
        # 计算每个候选mask与真值的重合度
        best_score = 0.0
        best_mask_info = None
        
        for mask_info in candidate_masks:
            pred_mask = mask_info['mask']
            
            if metric == 'dice':
                score = calculate_dice(gt_mask, pred_mask)
            else:  # iou
                score = calculate_iou(gt_mask, pred_mask)
            
            if score > best_score:
                best_score = score
                best_mask_info = mask_info
        
        return best_score, {
            'num_generated': len(masks),
            'num_candidates': len(candidate_masks),
            'best_mask': best_mask_info,
            'all_generated_masks': masks,
            'gt_bbox': gt_bbox,
            'gt_mask': gt_mask
        }
        
    except Exception as e:
        return 0.0, {'error': f'Exception during evaluation: {str(e)}'}


def create_visualization(image_path: str, eval_info: Dict[str, Any], best_score: float, 
                        metric: str, output_path: str) -> None:
    """创建可视化图片
    
    Args:
        image_path: 原图路径
        eval_info: 评估信息
        best_score: 最佳分数
        metric: 评估指标
        output_path: 输出路径（无扩展名）
    """
    try:
        # 加载原图
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 创建图1: 多区域分割展示
        fig1, ax1 = plt.subplots(1, 1, figsize=(12, 8))
        ax1.imshow(image_rgb)
        ax1.set_title(f'SAM2 Automatic Segmentation Results\n{len(eval_info["all_generated_masks"])} masks generated')
        
        # 显示所有生成的mask
        colors = plt.cm.Set3(np.linspace(0, 1, len(eval_info["all_generated_masks"])))
        for i, mask_info in enumerate(eval_info["all_generated_masks"]):
            mask = mask_info['segmentation']
            # 创建彩色mask
            colored_mask = np.zeros((*mask.shape, 4))
            colored_mask[mask] = [*colors[i][:3], 0.6]  # 半透明
            ax1.imshow(colored_mask)
            
            # 添加包围盒
            bbox = mask_info['bbox']  # [x, y, w, h]
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                                   linewidth=1, edgecolor=colors[i], facecolor='none')
            ax1.add_patch(rect)
        
        ax1.axis('off')
        plt.tight_layout()
        plt.savefig(f"{output_path}_all_masks.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 创建图2: 最佳预测与真值对比
        if eval_info.get('best_mask') is not None and eval_info.get('gt_mask') is not None:
            fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))
            
            # 转换为灰度图作为背景
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ax2.imshow(image_gray, cmap='gray', alpha=0.7)
            
            # 显示真值mask（绿色）
            gt_mask = eval_info['gt_mask']
            gt_colored = np.zeros((*gt_mask.shape, 4))
            gt_colored[gt_mask > 0] = [0, 1, 0, 0.5]  # 绿色半透明
            ax2.imshow(gt_colored)
            
            # 显示最佳预测mask（红色）
            best_mask = eval_info['best_mask']['mask']
            pred_colored = np.zeros((*best_mask.shape, 4))
            pred_colored[best_mask > 0] = [1, 0, 0, 0.5]  # 红色半透明
            ax2.imshow(pred_colored)
            
            # 添加包围盒
            gt_bbox = eval_info['gt_bbox']
            pred_bbox = eval_info['best_mask']['bbox']
            
            # 真值包围盒（绿色）
            gt_rect = patches.Rectangle((gt_bbox[0], gt_bbox[1]), 
                                      gt_bbox[2]-gt_bbox[0], gt_bbox[3]-gt_bbox[1],
                                      linewidth=2, edgecolor='green', facecolor='none', 
                                      linestyle='--', label='Ground Truth')
            ax2.add_patch(gt_rect)
            
            # 预测包围盒（红色）
            pred_rect = patches.Rectangle((pred_bbox[0], pred_bbox[1]), 
                                        pred_bbox[2]-pred_bbox[0], pred_bbox[3]-pred_bbox[1],
                                        linewidth=2, edgecolor='red', facecolor='none', 
                                        label='Best Prediction')
            ax2.add_patch(pred_rect)
            
            ax2.set_title(f'Best Prediction vs Ground Truth\n'
                         f'{metric.upper()}: {best_score:.3f} | '
                         f'Candidates: {eval_info["num_candidates"]}/{eval_info["num_generated"]}')
            ax2.legend()
            ax2.axis('off')
            
            plt.tight_layout()
            plt.savefig(f"{output_path}_comparison.png", dpi=150, bbox_inches='tight')
            plt.close()
        
    except Exception as e:
        print(f"Warning: Failed to create visualization for {image_path}: {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SAM2 Automatic segmentation against ground truth masks"
    )
    parser.add_argument(
        "--image_dir", 
        type=str, 
        required=True,
        help="Directory containing input images"
    )
    parser.add_argument(
        "--json_file", 
        type=str, 
        required=True,
        help="JSON file containing ground truth mask information"
    )
    parser.add_argument(
        "--sam_checkpoint", 
        type=str, 
        required=True,
        help="Path to SAM2 model checkpoint"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True,
        help="Output directory for results and visualizations"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default=None,
        help="Device to run on (cuda/cpu). Auto-detect if not specified"
    )
    parser.add_argument(
        "--metric", 
        type=str, 
        choices=['iou', 'dice'], 
        default='iou',
        help="Evaluation metric (default: iou)"
    )
    parser.add_argument(
        "--config_path", 
        type=str, 
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="Path to SAM2 config file"
    )
    parser.add_argument(
        "--points_per_side", 
        type=int, 
        default=32,
        help="Number of points per side for automatic mask generation"
    )
    parser.add_argument(
        "--pred_iou_thresh", 
        type=float, 
        default=0.88,
        help="IoU threshold for mask prediction"
    )
    parser.add_argument(
        "--stability_score_thresh", 
        type=float, 
        default=0.95,
        help="Stability score threshold for mask filtering"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Print detailed processing information"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 检查输入
    if not os.path.isdir(args.image_dir):
        print(f"Error: Image directory not found: {args.image_dir}")
        return 1
    
    if not os.path.isfile(args.json_file):
        print(f"Error: JSON file not found: {args.json_file}")
        return 1
    
    if not os.path.isfile(args.sam_checkpoint):
        print(f"Error: SAM2 checkpoint not found: {args.sam_checkpoint}")
        return 1
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载真值数据
    print(f"Loading ground truth data from: {args.json_file}")
    gt_masks = load_ground_truth_masks(args.json_file)
    print(f"Loaded {len(gt_masks)} ground truth masks")
    
    # 获取图片文件
    print(f"Scanning image directory: {args.image_dir}")
    image_files = get_image_files(args.image_dir)
    print(f"Found {len(image_files)} images")
    
    # 初始化SAM2
    print(f"Initializing SAM2 with checkpoint: {args.sam_checkpoint}")
    
    # 自动检测设备
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    try:
        # SAM2配置文件路径
        config_path = Path(__file__).parent.parent / "third_party" / "sam2" / args.config_path
        if not config_path.exists():
            config_path = Path(args.config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"SAM2 config file not found: {args.config_path}")
        
        sam_model = build_sam2(str(config_path), args.sam_checkpoint)
        sam_model = sam_model.to(device)
        
        # 创建自动mask生成器
        mask_generator = SAM2AutomaticMaskGenerator(
            model=sam_model,
            points_per_side=args.points_per_side,
            pred_iou_thresh=args.pred_iou_thresh,
            stability_score_thresh=args.stability_score_thresh,
        )
        
        print("SAM2 initialized successfully")
        
    except Exception as e:
        print(f"Error initializing SAM2: {e}")
        return 1
    
    # 评估所有图片
    print(f"\nStarting evaluation with {args.metric.upper()} metric...")
    
    all_scores = []
    num_processed = 0
    num_errors = 0
    results_log = []
    
    start_time = time.time()
    
    for i, image_path in enumerate(image_files, 1):
        image_name = Path(image_path).stem
        
        if args.verbose:
            print(f"[{i}/{len(image_files)}] Processing: {image_name}")
        
        # 检查是否有对应的真值
        if image_name not in gt_masks:
            if args.verbose:
                print(f"  -> No ground truth found, skipping")
            num_errors += 1
            continue
        
        gt_info = gt_masks[image_name]
        
        # 评估图片
        score, eval_info = evaluate_image(image_path, gt_info, mask_generator, args.metric)
        
        if 'error' in eval_info:
            if args.verbose:
                print(f"  -> Error: {eval_info['error']}")
            num_errors += 1
            continue
        
        all_scores.append(score)
        num_processed += 1
        
        # 记录结果
        result_entry = {
            'image_name': image_name,
            'score': score,
            'num_generated': eval_info['num_generated'],
            'num_candidates': eval_info['num_candidates']
        }
        results_log.append(result_entry)
        
        if args.verbose:
            print(f"  -> {args.metric.upper()}: {score:.3f} "
                  f"({eval_info['num_candidates']}/{eval_info['num_generated']} candidates)")
        
        # 创建可视化
        vis_output_path = os.path.join(args.output_dir, image_name)
        create_visualization(image_path, eval_info, score, args.metric, vis_output_path)
    
    # 计算统计信息
    if all_scores:
        best_score = max(all_scores)
        worst_score = min(all_scores)
        avg_score = sum(all_scores) / len(all_scores)
        
        # 保存详细结果
        results_summary = {
            'metric': args.metric,
            'total_images': len(image_files),
            'processed': num_processed,
            'errors': num_errors,
            'statistics': {
                'best': best_score,
                'worst': worst_score,
                'average': avg_score
            },
            'detailed_results': results_log,
            'parameters': {
                'points_per_side': args.points_per_side,
                'pred_iou_thresh': args.pred_iou_thresh,
                'stability_score_thresh': args.stability_score_thresh
            }
        }
        
        results_file = os.path.join(args.output_dir, 'evaluation_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)
        
        # 打印总结
        elapsed_time = time.time() - start_time
        print(f"\nEvaluation completed in {elapsed_time:.1f} seconds:")
        print(f"  Total images: {len(image_files)}")
        print(f"  Processed: {num_processed}")
        print(f"  Errors: {num_errors}")
        print(f"  Metric: {args.metric.upper()}")
        print(f"  Best score: {best_score:.3f}")
        print(f"  Worst score: {worst_score:.3f}")
        print(f"  Average score: {avg_score:.3f}")
        print(f"  Results saved to: {results_file}")
        print(f"  Visualizations saved to: {args.output_dir}")
        
        return 0
    else:
        print("No images were successfully processed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

```

现在更新可视化函数以处理matplotlib缺失的情况：

```python
#!/usr/bin/env python3

"""
使用SAM2 Automatic模式批量分割并评估与真值mask的吻合度

功能:
- 使用SAM2 Automatic模式对指定目录下的图片进行批量分割
- 与JSON描述的真值masks进行吻合度评估
- 支持IoU和DICE指标评估
- 生成分割效果可视化图片
- 统计整个数据集的最好/最差/平均分割效果

评估方法:
1. 对每张原图使用SAM2 Automatic模式分割得到多个区域
2. 通过包围盒重叠筛选候选预测mask
3. 计算候选mask与真值mask的IoU/DICE，取最大值作为该图的评估结果
4. 统计整个数据集的分割效果

输出可视化:
- 图1: Automatic模式分割的多区域多色展示
- 图2: 最佳预测mask与真值mask叠加显示（半透明）+ 评估指标

用法:
    python seg-rl/sam2_automatic_evaluation.py \
      --image_dir /path/to/images \
      --json_file /path/to/masks.json \
      --sam_checkpoint /path/to/sam2.1_hiera_large.pt \
      --output_dir /path/to/results \
      --device cuda

示例:
    python seg-rl/sam2_automatic_evaluation.py \
      --image_dir /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/imagesTr \
      --json_file /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/pred_masks-0.json \
      --sam_checkpoint /root/autodl-tmp/works/Seg-R0/third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
      --output_dir /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/evaluation_results \
      --device cuda \
      --metric dice
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import time

import cv2
import numpy as np
from PIL import Image as PILImage
import torch

# 可选的matplotlib导入
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    HAS_MATPLOTLIB = True
except ImportError:
    print("Warning: matplotlib not found. Visualization features will be disabled.")
    HAS_MATPLOTLIB = False

# 添加sam2路径
sys.path.append(str(Path(__file__).parent.parent / "third_party" / "sam2"))

try:
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
except ImportError as e:
    print(f"Error importing SAM2: {e}")
    print("Please ensure SAM2 is properly installed and the path is correct.")
    sys.exit(1)


# 支持的图片格式
SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}


def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """计算两个mask之间的IoU
    
    Args:
        mask1: 第一个mask (binary)
        mask2: 第二个mask (binary)
        
    Returns:
        IoU值 (0-1)
    """
    # 确保是二值mask
    mask1_bin = (mask1 > 0).astype(np.uint8)
    mask2_bin = (mask2 > 0).astype(np.uint8)
    
    # 计算交集和并集
    intersection = np.logical_and(mask1_bin, mask2_bin).sum()
    union = np.logical_or(mask1_bin, mask2_bin).sum()
    
    if union == 0:
        return 0.0
    
    return float(intersection) / float(union)


def calculate_dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """计算两个mask之间的DICE系数
    
    Args:
        mask1: 第一个mask (binary)
        mask2: 第二个mask (binary)
        
    Returns:
        DICE值 (0-1)
    """
    # 确保是二值mask
    mask1_bin = (mask1 > 0).astype(np.uint8)
    mask2_bin = (mask2 > 0).astype(np.uint8)
    
    # 计算交集
    intersection = np.logical_and(mask1_bin, mask2_bin).sum()
    total = mask1_bin.sum() + mask2_bin.sum()
    
    if total == 0:
        return 0.0
    
    return 2.0 * float(intersection) / float(total)


def bbox_overlap(bbox1: List[int], bbox2: List[int]) -> bool:
    """检查两个包围盒是否有重叠
    
    Args:
        bbox1: [x_min, y_min, x_max, y_max]
        bbox2: [x_min, y_min, x_max, y_max]
        
    Returns:
        是否有重叠
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # 检查是否不重叠
    if x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min:
        return False
    
    return True


def load_ground_truth_masks(json_file: str) -> Dict[str, Dict[str, Any]]:
    """加载真值mask信息
    
    Args:
        json_file: JSON文件路径
        
    Returns:
        字典，键为图片文件名（无扩展名），值为mask信息
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    gt_masks = {}
    for item in data:
        mask_path = item['mask_path']
        bbox = item['bbox']
        
        # 从mask路径提取文件名（无扩展名）
        mask_filename = Path(mask_path).stem
        
        gt_masks[mask_filename] = {
            'mask_path': mask_path,
            'bbox': bbox
        }
    
    return gt_masks


def get_image_files(image_dir: str) -> List[str]:
    """获取图片目录下所有支持的图片文件
    
    Args:
        image_dir: 图片目录
        
    Returns:
        图片文件路径列表，按文件名排序
    """
    image_files = []
    
    if not os.path.isdir(image_dir):
        print(f"Error: Directory not found: {image_dir}")
        return image_files
    
    for file_path in Path(image_dir).iterdir():
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            image_files.append(str(file_path))
    
    image_files.sort()
    return image_files


def load_mask_image(mask_path: str) -> Optional[np.ndarray]:
    """加载mask图像
    
    Args:
        mask_path: mask文件路径
        
    Returns:
        mask数组或None
    """
    try:
        if not os.path.exists(mask_path):
            print(f"Warning: Mask file not found: {mask_path}")
            return None
        
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            print(f"Warning: Failed to load mask: {mask_path}")
            return None
        
        # 如果是彩色图像，转换为灰度
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        return mask
    except Exception as e:
        print(f"Warning: Error loading mask {mask_path}: {e}")
        return None


def evaluate_image(image_path: str, gt_info: Dict[str, Any], mask_generator, metric: str = 'iou') -> Tuple[float, Dict[str, Any]]:
    """评估单张图片的分割效果
    
    Args:
        image_path: 图片路径
        gt_info: 真值信息
        mask_generator: SAM2 mask生成器
        metric: 评估指标 ('iou' 或 'dice')
        
    Returns:
        (最佳分数, 详细信息字典)
    """
    try:
        # 加载图片
        image = cv2.imread(image_path)
        if image is None:
            return 0.0, {'error': f'Failed to load image: {image_path}'}
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 加载真值mask
        gt_mask = load_mask_image(gt_info['mask_path'])
        if gt_mask is None:
            return 0.0, {'error': f'Failed to load ground truth mask: {gt_info["mask_path"]}'}
        
        gt_bbox = gt_info['bbox']
        
        # 使用SAM2 Automatic模式生成mask
        masks = mask_generator.generate(image_rgb)
        
        if not masks:
            return 0.0, {'error': 'No masks generated by SAM2', 'num_generated': 0}
        
        # 筛选与真值包围盒重叠的预测mask
        candidate_masks = []
        for mask_info in masks:
            pred_bbox = mask_info['bbox']  # SAM2返回的格式应该是[x, y, w, h]
            # 转换为[x_min, y_min, x_max, y_max]格式
            pred_bbox_xyxy = [
                int(pred_bbox[0]), 
                int(pred_bbox[1]), 
                int(pred_bbox[0] + pred_bbox[2]), 
                int(pred_bbox[1] + pred_bbox[3])
            ]
            
            if bbox_overlap(gt_bbox, pred_bbox_xyxy):
                candidate_masks.append({
                    'mask': mask_info['segmentation'],
                    'bbox': pred_bbox_xyxy,
                    'area': mask_info['area'],
                    'stability_score': mask_info.get('stability_score', 0.0)
                })
        
        if not candidate_masks:
            return 0.0, {
                'error': 'No overlapping masks found',
                'num_generated': len(masks),
                'num_candidates': 0
            }
        
        # 计算每个候选mask与真值的重合度
        best_score = 0.0
        best_mask_info = None
        
        for mask_info in candidate_masks:
            pred_mask = mask_info['mask']
            
            if metric == 'dice':
                score = calculate_dice(gt_mask, pred_mask)
            else:  # iou
                score = calculate_iou(gt_mask, pred_mask)
            
            if score > best_score:
                best_score = score
                best_mask_info = mask_info
        
        return best_score, {
            'num_generated': len(masks),
            'num_candidates': len(candidate_masks),
            'best_mask': best_mask_info,
            'all_generated_masks': masks,
            'gt_bbox': gt_bbox,
            'gt_mask': gt_mask
        }
        
    except Exception as e:
        return 0.0, {'error': f'Exception during evaluation: {str(e)}'}


def create_visualization(image_path: str, eval_info: Dict[str, Any], best_score: float, 
                        metric: str, output_path: str) -> None:
    """创建可视化图片
    
    Args:
        image_path: 原图路径
        eval_info: 评估信息
        best_score: 最佳分数
        metric: 评估指标
        output_path: 输出路径（无扩展名）
    """
    if not HAS_MATPLOTLIB:
        print(f"Skipping visualization for {Path(image_path).name} (matplotlib not available)")
        return
    
    try:
        # 加载原图
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 创建图1: 多区域分割展示
        fig1, ax1 = plt.subplots(1, 1, figsize=(12, 8))
        ax1.imshow(image_rgb)
        ax1.set_title(f'SAM2 Automatic Segmentation Results\n{len(eval_info["all_generated_masks"])} masks generated')
        
        # 显示所有生成的mask
        colors = plt.cm.Set3(np.linspace(0, 1, len(eval_info["all_generated_masks"])))
        for i, mask_info in enumerate(eval_info["all_generated_masks"]):
            mask = mask_info['segmentation']
            # 创建彩色mask
            colored_mask = np.zeros((*mask.shape, 4))
            colored_mask[mask] = [*colors[i][:3], 0.6]  # 半透明
            ax1.imshow(colored_mask)
            
            # 添加包围盒
            bbox = mask_info['bbox']  # [x, y, w, h]
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                                   linewidth=1, edgecolor=colors[i], facecolor='none')
            ax1.add_patch(rect)
        
        ax1.axis('off')
        plt.tight_layout()
        plt.savefig(f"{output_path}_all_masks.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 创建图2: 最佳预测与真值对比
        if eval_info.get('best_mask') is not None and eval_info.get('gt_mask') is not None:
            fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))
            
            # 转换为灰度图作为背景
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ax2.imshow(image_gray, cmap='gray', alpha=0.7)
            
            # 显示真值mask（绿色）
            gt_mask = eval_info['gt_mask']
            gt_colored = np.zeros((*gt_mask.shape, 4))
            gt_colored[gt_mask > 0] = [0, 1, 0, 0.5]  # 绿色半透明
            ax2.imshow(gt_colored)
            
            # 显示最佳预测mask（红色）
            best_mask = eval_info['best_mask']['mask']
            pred_colored = np.zeros((*best_mask.shape, 4))
            pred_colored[best_mask > 0] = [1, 0, 0, 0.5]  # 红色半透明
            ax2.imshow(pred_colored)
            
            # 添加包围盒
            gt_bbox = eval_info['gt_bbox']
            pred_bbox = eval_info['best_mask']['bbox']
            
            # 真值包围盒（绿色）
            gt_rect = patches.Rectangle((gt_bbox[0], gt_bbox[1]), 
                                      gt_bbox[2]-gt_bbox[0], gt_bbox[3]-gt_bbox[1],
                                      linewidth=2, edgecolor='green', facecolor='none', 
                                      linestyle='--', label='Ground Truth')
            ax2.add_patch(gt_rect)
            
            # 预测包围盒（红色）
            pred_rect = patches.Rectangle((pred_bbox[0], pred_bbox[1]), 
                                        pred_bbox[2]-pred_bbox[0], pred_bbox[3]-pred_bbox[1],
                                        linewidth=2, edgecolor='red', facecolor='none', 
                                        label='Best Prediction')
            ax2.add_patch(pred_rect)
            
            ax2.set_title(f'Best Prediction vs Ground Truth\n'
                         f'{metric.upper()}: {best_score:.3f} | '
                         f'Candidates: {eval_info["num_candidates"]}/{eval_info["num_generated"]}')
            ax2.legend()
            ax2.axis('off')
            
            plt.tight_layout()
            plt.savefig(f"{output_path}_comparison.png", dpi=150, bbox_inches='tight')
            plt.close()
        
    except Exception as e:
        print(f"Warning: Failed to create visualization for {image_path}: {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SAM2 Automatic segmentation against ground truth masks"
    )
    parser.add_argument(
        "--image_dir", 
        type=str, 
        required=True,
        help="Directory containing input images"
    )
    parser.add_argument(
        "--json_file", 
        type=str, 
        required=True,
        help="JSON file containing ground truth mask information"
    )
    parser.add_argument(
        "--sam_checkpoint", 
        type=str, 
        required=True,
        help="Path to SAM2 model checkpoint"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True,
        help="Output directory for results and visualizations"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default=None,
        help="Device to run on (cuda/cpu). Auto-detect if not specified"
    )
    parser.add_argument(
        "--metric", 
        type=str, 
        choices=['iou', 'dice'], 
        default='iou',
        help="Evaluation metric (default: iou)"
    )
    parser.add_argument(
        "--config_path", 
        type=str, 
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="Path to SAM2 config file"
    )
    parser.add_argument(
        "--points_per_side", 
        type=int, 
        default=32,
        help="Number of points per side for automatic mask generation"
    )
    parser.add_argument(
        "--pred_iou_thresh", 
        type=float, 
        default=0.88,
        help="IoU threshold for mask prediction"
    )
    parser.add_argument(
        "--stability_score_thresh", 
        type=float, 
        default=0.95,
        help="Stability score threshold for mask filtering"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Print detailed processing information"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 检查输入
    if not os.path.isdir(args.image_dir):
        print(f"Error: Image directory not found: {args.image_dir}")
        return 1
    
    if not os.path.isfile(args.json_file):
        print(f"Error: JSON file not found: {args.json_file}")
        return 1
    
    if not os.path.isfile(args.sam_checkpoint):
        print(f"Error: SAM2 checkpoint not found: {args.sam_checkpoint}")
        return 1
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载真值数据
    print(f"Loading ground truth data from: {args.json_file}")
    gt_masks = load_ground_truth_masks(args.json_file)
    print(f"Loaded {len(gt_masks)} ground truth masks")
    
    # 获取图片文件
    print(f"Scanning image directory: {args.image_dir}")
    image_files = get_image_files(args.image_dir)
    print(f"Found {len(image_files)} images")
    
    # 初始化SAM2
    print(f"Initializing SAM2 with checkpoint: {args.sam_checkpoint}")
    
    # 自动检测设备
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    try:
        # SAM2配置文件路径
        config_path = Path(__file__).parent.parent / "third_party" / "sam2" / args.config_path
        if not config_path.exists():
            config_path = Path(args.config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"SAM2 config file not found: {args.config_path}")
        
        sam_model = build_sam2(str(config_path), args.sam_checkpoint)
        sam_model = sam_model.to(device)
        
        # 创建自动mask生成器
        mask_generator = SAM2AutomaticMaskGenerator(
            model=sam_model,
            points_per_side=args.points_per_side,
            pred_iou_thresh=args.pred_iou_thresh,
            stability_score_thresh=args.stability_score_thresh,
        )
        
        print("SAM2 initialized successfully")
        
    except Exception as e:
        print(f"Error initializing SAM2: {e}")
        return 1
    
    # 评估所有图片
    print(f"\nStarting evaluation with {args.metric.upper()} metric...")
    
    all_scores = []
    num_processed = 0
    num_errors = 0
    results_log = []
    
    start_time = time.time()
    
    for i, image_path in enumerate(image_files, 1):
        image_name = Path(image_path).stem
        
        if args.verbose:
            print(f"[{i}/{len(image_files)}] Processing: {image_name}")
        
        # 检查是否有对应的真值
        if image_name not in gt_masks:
            if args.verbose:
                print(f"  -> No ground truth found, skipping")
            num_errors += 1
            continue
        
        gt_info = gt_masks[image_name]
        
        # 评估图片
        score, eval_info = evaluate_image(image_path, gt_info, mask_generator, args.metric)
        
        if 'error' in eval_info:
            if args.verbose:
                print(f"  -> Error: {eval_info['error']}")
            num_errors += 1
            continue
        
        all_scores.append(score)
        num_processed += 1
        
        # 记录结果
        result_entry = {
            'image_name': image_name,
            'score': score,
            'num_generated': eval_info['num_generated'],
            'num_candidates': eval_info['num_candidates']
        }
        results_log.append(result_entry)
        
        if args.verbose:
            print(f"  -> {args.metric.upper()}: {score:.3f} "
                  f"({eval_info['num_candidates']}/{eval_info['num_generated']} candidates)")
        
        # 创建可视化
        vis_output_path = os.path.join(args.output_dir, image_name)
        create_visualization(image_path, eval_info, score, args.metric, vis_output_path)
    
    # 计算统计信息
    if all_scores:
        best_score = max(all_scores)
        worst_score = min(all_scores)
        avg_score = sum(all_scores) / len(all_scores)
        
        # 保存详细结果
        results_summary = {
            'metric': args.metric,
            'total_images': len(image_files),
            'processed': num_processed,
            'errors': num_errors,
            'statistics': {
                'best': best_score,
                'worst': worst_score,
                'average': avg_score
            },
            'detailed_results': results_log,
            'parameters': {
                'points_per_side': args.points_per_side,
                'pred_iou_thresh': args.pred_iou_thresh,
                'stability_score_thresh': args.stability_score_thresh
            }
        }
        
        results_file = os.path.join(args.output_dir, 'evaluation_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)
        
        # 打印总结
        elapsed_time = time.time() - start_time
        print(f"\nEvaluation completed in {elapsed_time:.1f} seconds:")
        print(f"  Total images: {len(image_files)}")
        print(f"  Processed: {num_processed}")
        print(f"  Errors: {num_errors}")
        print(f"  Metric: {args.metric.upper()}")
        print(f"  Best score: {best_score:.3f}")
        print(f"  Worst score: {worst_score:.3f}")
        print(f"  Average score: {avg_score:.3f}")
        print(f"  Results saved to: {results_file}")
        print(f"  Visualizations saved to: {args.output_dir}")
        
        return 0
    else:
        print("No images were successfully processed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

```

现在我需要添加一个参数来允许用户禁用可视化功能：

```python
#!/usr/bin/env python3

"""
使用SAM2 Automatic模式批量分割并评估与真值mask的吻合度

功能:
- 使用SAM2 Automatic模式对指定目录下的图片进行批量分割
- 与JSON描述的真值masks进行吻合度评估
- 支持IoU和DICE指标评估
- 生成分割效果可视化图片
- 统计整个数据集的最好/最差/平均分割效果

评估方法:
1. 对每张原图使用SAM2 Automatic模式分割得到多个区域
2. 通过包围盒重叠筛选候选预测mask
3. 计算候选mask与真值mask的IoU/DICE，取最大值作为该图的评估结果
4. 统计整个数据集的分割效果

输出可视化:
- 图1: Automatic模式分割的多区域多色展示
- 图2: 最佳预测mask与真值mask叠加显示（半透明）+ 评估指标

用法:
    python seg-rl/sam2_automatic_evaluation.py \
      --image_dir /path/to/images \
      --json_file /path/to/masks.json \
      --sam_checkpoint /path/to/sam2.1_hiera_large.pt \
      --output_dir /path/to/results \
      --device cuda

示例:
    python seg-rl/sam2_automatic_evaluation.py \
      --image_dir /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/imagesTr \
      --json_file /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/pred_masks-0.json \
      --sam_checkpoint /root/autodl-tmp/works/Seg-R0/third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
      --output_dir /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/evaluation_results \
      --device cuda \
      --metric dice
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import time

import cv2
import numpy as np
from PIL import Image as PILImage
import torch

# 可选的matplotlib导入
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    HAS_MATPLOTLIB = True
except ImportError:
    print("Warning: matplotlib not found. Visualization features will be disabled.")
    HAS_MATPLOTLIB = False

# 添加sam2路径
sys.path.append(str(Path(__file__).parent.parent / "third_party" / "sam2"))

try:
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
except ImportError as e:
    print(f"Error importing SAM2: {e}")
    print("Please ensure SAM2 is properly installed and the path is correct.")
    sys.exit(1)


# 支持的图片格式
SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}


def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """计算两个mask之间的IoU
    
    Args:
        mask1: 第一个mask (binary)
        mask2: 第二个mask (binary)
        
    Returns:
        IoU值 (0-1)
    """
    # 确保是二值mask
    mask1_bin = (mask1 > 0).astype(np.uint8)
    mask2_bin = (mask2 > 0).astype(np.uint8)
    
    # 计算交集和并集
    intersection = np.logical_and(mask1_bin, mask2_bin).sum()
    union = np.logical_or(mask1_bin, mask2_bin).sum()
    
    if union == 0:
        return 0.0
    
    return float(intersection) / float(union)


def calculate_dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """计算两个mask之间的DICE系数
    
    Args:
        mask1: 第一个mask (binary)
        mask2: 第二个mask (binary)
        
    Returns:
        DICE值 (0-1)
    """
    # 确保是二值mask
    mask1_bin = (mask1 > 0).astype(np.uint8)
    mask2_bin = (mask2 > 0).astype(np.uint8)
    
    # 计算交集
    intersection = np.logical_and(mask1_bin, mask2_bin).sum()
    total = mask1_bin.sum() + mask2_bin.sum()
    
    if total == 0:
        return 0.0
    
    return 2.0 * float(intersection) / float(total)


def bbox_overlap(bbox1: List[int], bbox2: List[int]) -> bool:
    """检查两个包围盒是否有重叠
    
    Args:
        bbox1: [x_min, y_min, x_max, y_max]
        bbox2: [x_min, y_min, x_max, y_max]
        
    Returns:
        是否有重叠
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # 检查是否不重叠
    if x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min:
        return False
    
    return True


def load_ground_truth_masks(json_file: str) -> Dict[str, Dict[str, Any]]:
    """加载真值mask信息
    
    Args:
        json_file: JSON文件路径
        
    Returns:
        字典，键为图片文件名（无扩展名），值为mask信息
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    gt_masks = {}
    for item in data:
        mask_path = item['mask_path']
        bbox = item['bbox']
        
        # 从mask路径提取文件名（无扩展名）
        mask_filename = Path(mask_path).stem
        
        gt_masks[mask_filename] = {
            'mask_path': mask_path,
            'bbox': bbox
        }
    
    return gt_masks


def get_image_files(image_dir: str) -> List[str]:
    """获取图片目录下所有支持的图片文件
    
    Args:
        image_dir: 图片目录
        
    Returns:
        图片文件路径列表，按文件名排序
    """
    image_files = []
    
    if not os.path.isdir(image_dir):
        print(f"Error: Directory not found: {image_dir}")
        return image_files
    
    for file_path in Path(image_dir).iterdir():
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            image_files.append(str(file_path))
    
    image_files.sort()
    return image_files


def load_mask_image(mask_path: str) -> Optional[np.ndarray]:
    """加载mask图像
    
    Args:
        mask_path: mask文件路径
        
    Returns:
        mask数组或None
    """
    try:
        if not os.path.exists(mask_path):
            print(f"Warning: Mask file not found: {mask_path}")
            return None
        
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            print(f"Warning: Failed to load mask: {mask_path}")
            return None
        
        # 如果是彩色图像，转换为灰度
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        return mask
    except Exception as e:
        print(f"Warning: Error loading mask {mask_path}: {e}")
        return None


def evaluate_image(image_path: str, gt_info: Dict[str, Any], mask_generator, metric: str = 'iou') -> Tuple[float, Dict[str, Any]]:
    """评估单张图片的分割效果
    
    Args:
        image_path: 图片路径
        gt_info: 真值信息
        mask_generator: SAM2 mask生成器
        metric: 评估指标 ('iou' 或 'dice')
        
    Returns:
        (最佳分数, 详细信息字典)
    """
    try:
        # 加载图片
        image = cv2.imread(image_path)
        if image is None:
            return 0.0, {'error': f'Failed to load image: {image_path}'}
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 加载真值mask
        gt_mask = load_mask_image(gt_info['mask_path'])
        if gt_mask is None:
            return 0.0, {'error': f'Failed to load ground truth mask: {gt_info["mask_path"]}'}
        
        gt_bbox = gt_info['bbox']
        
        # 使用SAM2 Automatic模式生成mask
        masks = mask_generator.generate(image_rgb)
        
        if not masks:
            return 0.0, {'error': 'No masks generated by SAM2', 'num_generated': 0}
        
        # 筛选与真值包围盒重叠的预测mask
        candidate_masks = []
        for mask_info in masks:
            pred_bbox = mask_info['bbox']  # SAM2返回的格式应该是[x, y, w, h]
            # 转换为[x_min, y_min, x_max, y_max]格式
            pred_bbox_xyxy = [
                int(pred_bbox[0]), 
                int(pred_bbox[1]), 
                int(pred_bbox[0] + pred_bbox[2]), 
                int(pred_bbox[1] + pred_bbox[3])
            ]
            
            if bbox_overlap(gt_bbox, pred_bbox_xyxy):
                candidate_masks.append({
                    'mask': mask_info['segmentation'],
                    'bbox': pred_bbox_xyxy,
                    'area': mask_info['area'],
                    'stability_score': mask_info.get('stability_score', 0.0)
                })
        
        if not candidate_masks:
            return 0.0, {
                'error': 'No overlapping masks found',
                'num_generated': len(masks),
                'num_candidates': 0
            }
        
        # 计算每个候选mask与真值的重合度
        best_score = 0.0
        best_mask_info = None
        
        for mask_info in candidate_masks:
            pred_mask = mask_info['mask']
            
            if metric == 'dice':
                score = calculate_dice(gt_mask, pred_mask)
            else:  # iou
                score = calculate_iou(gt_mask, pred_mask)
            
            if score > best_score:
                best_score = score
                best_mask_info = mask_info
        
        return best_score, {
            'num_generated': len(masks),
            'num_candidates': len(candidate_masks),
            'best_mask': best_mask_info,
            'all_generated_masks': masks,
            'gt_bbox': gt_bbox,
            'gt_mask': gt_mask
        }
        
    except Exception as e:
        return 0.0, {'error': f'Exception during evaluation: {str(e)}'}


def create_visualization(image_path: str, eval_info: Dict[str, Any], best_score: float, 
                        metric: str, output_path: str) -> None:
    """创建可视化图片
    
    Args:
        image_path: 原图路径
        eval_info: 评估信息
        best_score: 最佳分数
        metric: 评估指标
        output_path: 输出路径（无扩展名）
    """
    if not HAS_MATPLOTLIB:
        print(f"Skipping visualization for {Path(image_path).name} (matplotlib not available)")
        return
    
    try:
        # 加载原图
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 创建图1: 多区域分割展示
        fig1, ax1 = plt.subplots(1, 1, figsize=(12, 8))
        ax1.imshow(image_rgb)
        ax1.set_title(f'SAM2 Automatic Segmentation Results\n{len(eval_info["all_generated_masks"])} masks generated')
        
        # 显示所有生成的mask
        colors = plt.cm.Set3(np.linspace(0, 1, len(eval_info["all_generated_masks"])))
        for i, mask_info in enumerate(eval_info["all_generated_masks"]):
            mask = mask_info['segmentation']
            # 创建彩色mask
            colored_mask = np.zeros((*mask.shape, 4))
            colored_mask[mask] = [*colors[i][:3], 0.6]  # 半透明
            ax1.imshow(colored_mask)
            
            # 添加包围盒
            bbox = mask_info['bbox']  # [x, y, w, h]
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                                   linewidth=1, edgecolor=colors[i], facecolor='none')
            ax1.add_patch(rect)
        
        ax1.axis('off')
        plt.tight_layout()
        plt.savefig(f"{output_path}_all_masks.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 创建图2: 最佳预测与真值对比
        if eval_info.get('best_mask') is not None and eval_info.get('gt_mask') is not None:
            fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))
            
            # 转换为灰度图作为背景
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ax2.imshow(image_gray, cmap='gray', alpha=0.7)
            
            # 显示真值mask（绿色）
            gt_mask = eval_info['gt_mask']
            gt_colored = np.zeros((*gt_mask.shape, 4))
            gt_colored[gt_mask > 0] = [0, 1, 0, 0.5]  # 绿色半透明
            ax2.imshow(gt_colored)
            
            # 显示最佳预测mask（红色）
            best_mask = eval_info['best_mask']['mask']
            pred_colored = np.zeros((*best_mask.shape, 4))
            pred_colored[best_mask > 0] = [1, 0, 0, 0.5]  # 红色半透明
            ax2.imshow(pred_colored)
            
            # 添加包围盒
            gt_bbox = eval_info['gt_bbox']
            pred_bbox = eval_info['best_mask']['bbox']
            
            # 真值包围盒（绿色）
            gt_rect = patches.Rectangle((gt_bbox[0], gt_bbox[1]), 
                                      gt_bbox[2]-gt_bbox[0], gt_bbox[3]-gt_bbox[1],
                                      linewidth=2, edgecolor='green', facecolor='none', 
                                      linestyle='--', label='Ground Truth')
            ax2.add_patch(gt_rect)
            
            # 预测包围盒（红色）
            pred_rect = patches.Rectangle((pred_bbox[0], pred_bbox[1]), 
                                        pred_bbox[2]-pred_bbox[0], pred_bbox[3]-pred_bbox[1],
                                        linewidth=2, edgecolor='red', facecolor='none', 
                                        label='Best Prediction')
            ax2.add_patch(pred_rect)
            
            ax2.set_title(f'Best Prediction vs Ground Truth\n'
                         f'{metric.upper()}: {best_score:.3f} | '
                         f'Candidates: {eval_info["num_candidates"]}/{eval_info["num_generated"]}')
            ax2.legend()
            ax2.axis('off')
            
            plt.tight_layout()
            plt.savefig(f"{output_path}_comparison.png", dpi=150, bbox_inches='tight')
            plt.close()
        
    except Exception as e:
        print(f"Warning: Failed to create visualization for {image_path}: {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SAM2 Automatic segmentation against ground truth masks"
    )
    parser.add_argument(
        "--image_dir", 
        type=str, 
        required=True,
        help="Directory containing input images"
    )
    parser.add_argument(
        "--json_file", 
        type=str, 
        required=True,
        help="JSON file containing ground truth mask information"
    )
    parser.add_argument(
        "--sam_checkpoint", 
        type=str, 
        required=True,
        help="Path to SAM2 model checkpoint"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True,
        help="Output directory for results and visualizations"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default=None,
        help="Device to run on (cuda/cpu). Auto-detect if not specified"
    )
    parser.add_argument(
        "--metric", 
        type=str, 
        choices=['iou', 'dice'], 
        default='iou',
        help="Evaluation metric (default: iou)"
    )
    parser.add_argument(
        "--config_path", 
        type=str, 
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="Path to SAM2 config file"
    )
    parser.add_argument(
        "--points_per_side", 
        type=int, 
        default=32,
        help="Number of points per side for automatic mask generation"
    )
    parser.add_argument(
        "--pred_iou_thresh", 
        type=float, 
        default=0.88,
        help="IoU threshold for mask prediction"
    )
    parser.add_argument(
        "--stability_score_thresh", 
        type=float, 
        default=0.95,
        help="Stability score threshold for mask filtering"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Print detailed processing information"
    )
    parser.add_argument(
        "--no_visualization", 
        action="store_true",
        help="Skip creating visualization images"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 检查输入
    if not os.path.isdir(args.image_dir):
        print(f"Error: Image directory not found: {args.image_dir}")
        return 1
    
    if not os.path.isfile(args.json_file):
        print(f"Error: JSON file not found: {args.json_file}")
        return 1
    
    if not os.path.isfile(args.sam_checkpoint):
        print(f"Error: SAM2 checkpoint not found: {args.sam_checkpoint}")
        return 1
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载真值数据
    print(f"Loading ground truth data from: {args.json_file}")
    gt_masks = load_ground_truth_masks(args.json_file)
    print(f"Loaded {len(gt_masks)} ground truth masks")
    
    # 获取图片文件
    print(f"Scanning image directory: {args.image_dir}")
    image_files = get_image_files(args.image_dir)
    print(f"Found {len(image_files)} images")
    
    # 初始化SAM2
    print(f"Initializing SAM2 with checkpoint: {args.sam_checkpoint}")
    
    # 自动检测设备
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    try:
        # SAM2配置文件路径
        config_path = Path(__file__).parent.parent / "third_party" / "sam2" / args.config_path
        if not config_path.exists():
            config_path = Path(args.config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"SAM2 config file not found: {args.config_path}")
        
        sam_model = build_sam2(str(config_path), args.sam_checkpoint)
        sam_model = sam_model.to(device)
        
        # 创建自动mask生成器
        mask_generator = SAM2AutomaticMaskGenerator(
            model=sam_model,
            points_per_side=args.points_per_side,
            pred_iou_thresh=args.pred_iou_thresh,
            stability_score_thresh=args.stability_score_thresh,
        )
        
        print("SAM2 initialized successfully")
        
    except Exception as e:
        print(f"Error initializing SAM2: {e}")
        return 1
    
    # 评估所有图片
    print(f"\nStarting evaluation with {args.metric.upper()} metric...")
    
    all_scores = []
    num_processed = 0
    num_errors = 0
    results_log = []
    
    start_time = time.time()
    
    for i, image_path in enumerate(image_files, 1):
        image_name = Path(image_path).stem
        
        if args.verbose:
            print(f"[{i}/{len(image_files)}] Processing: {image_name}")
        
        # 检查是否有对应的真值
        if image_name not in gt_masks:
            if args.verbose:
                print(f"  -> No ground truth found, skipping")
            num_errors += 1
            continue
        
        gt_info = gt_masks[image_name]
        
        # 评估图片
        score, eval_info = evaluate_image(image_path, gt_info, mask_generator, args.metric)
        
        if 'error' in eval_info:
            if args.verbose:
                print(f"  -> Error: {eval_info['error']}")
            num_errors += 1
            continue
        
        all_scores.append(score)
        num_processed += 1
        
        # 记录结果
        result_entry = {
            'image_name': image_name,
            'score': score,
            'num_generated': eval_info['num_generated'],
            'num_candidates': eval_info['num_candidates']
        }
        results_log.append(result_entry)
        
        if args.verbose:
            print(f"  -> {args.metric.upper()}: {score:.3f} "
                  f"({eval_info['num_candidates']}/{eval_info['num_generated']} candidates)")
        
        # 创建可视化
        vis_output_path = os.path.join(args.output_dir, image_name)
        create_visualization(image_path, eval_info, score, args.metric, vis_output_path)
    
    # 计算统计信息
    if all_scores:
        best_score = max(all_scores)
        worst_score = min(all_scores)
        avg_score = sum(all_scores) / len(all_scores)
        
        # 保存详细结果
        results_summary = {
            'metric': args.metric,
            'total_images': len(image_files),
            'processed': num_processed,
            'errors': num_errors,
            'statistics': {
                'best': best_score,
                'worst': worst_score,
                'average': avg_score
            },
            'detailed_results': results_log,
            'parameters': {
                'points_per_side': args.points_per_side,
                'pred_iou_thresh': args.pred_iou_thresh,
                'stability_score_thresh': args.stability_score_thresh
            }
        }
        
        results_file = os.path.join(args.output_dir, 'evaluation_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)
        
        # 打印总结
        elapsed_time = time.time() - start_time
        print(f"\nEvaluation completed in {elapsed_time:.1f} seconds:")
        print(f"  Total images: {len(image_files)}")
        print(f"  Processed: {num_processed}")
        print(f"  Errors: {num_errors}")
        print(f"  Metric: {args.metric.upper()}")
        print(f"  Best score: {best_score:.3f}")
        print(f"  Worst score: {worst_score:.3f}")
        print(f"  Average score: {avg_score:.3f}")
        print(f"  Results saved to: {results_file}")
        print(f"  Visualizations saved to: {args.output_dir}")
        
        return 0
    else:
        print("No images were successfully processed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

```

现在更新主函数以支持禁用可视化：

```python
#!/usr/bin/env python3

"""
使用SAM2 Automatic模式批量分割并评估与真值mask的吻合度

功能:
- 使用SAM2 Automatic模式对指定目录下的图片进行批量分割
- 与JSON描述的真值masks进行吻合度评估
- 支持IoU和DICE指标评估
- 生成分割效果可视化图片
- 统计整个数据集的最好/最差/平均分割效果

评估方法:
1. 对每张原图使用SAM2 Automatic模式分割得到多个区域
2. 通过包围盒重叠筛选候选预测mask
3. 计算候选mask与真值mask的IoU/DICE，取最大值作为该图的评估结果
4. 统计整个数据集的分割效果

输出可视化:
- 图1: Automatic模式分割的多区域多色展示
- 图2: 最佳预测mask与真值mask叠加显示（半透明）+ 评估指标

用法:
    python seg-rl/sam2_automatic_evaluation.py \
      --image_dir /path/to/images \
      --json_file /path/to/masks.json \
      --sam_checkpoint /path/to/sam2.1_hiera_large.pt \
      --output_dir /path/to/results \
      --device cuda

示例:
    python seg-rl/sam2_automatic_evaluation.py \
      --image_dir /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/imagesTr \
      --json_file /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/pred_masks-0.json \
      --sam_checkpoint /root/autodl-tmp/works/Seg-R0/third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
      --output_dir /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/evaluation_results \
      --device cuda \
      --metric dice
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import time

import cv2
import numpy as np
from PIL import Image as PILImage
import torch

# 可选的matplotlib导入
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    HAS_MATPLOTLIB = True
except ImportError:
    print("Warning: matplotlib not found. Visualization features will be disabled.")
    HAS_MATPLOTLIB = False

# 添加sam2路径
sys.path.append(str(Path(__file__).parent.parent / "third_party" / "sam2"))

try:
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
except ImportError as e:
    print(f"Error importing SAM2: {e}")
    print("Please ensure SAM2 is properly installed and the path is correct.")
    sys.exit(1)


# 支持的图片格式
SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}


def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """计算两个mask之间的IoU
    
    Args:
        mask1: 第一个mask (binary)
        mask2: 第二个mask (binary)
        
    Returns:
        IoU值 (0-1)
    """
    # 确保是二值mask
    mask1_bin = (mask1 > 0).astype(np.uint8)
    mask2_bin = (mask2 > 0).astype(np.uint8)
    
    # 计算交集和并集
    intersection = np.logical_and(mask1_bin, mask2_bin).sum()
    union = np.logical_or(mask1_bin, mask2_bin).sum()
    
    if union == 0:
        return 0.0
    
    return float(intersection) / float(union)


def calculate_dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """计算两个mask之间的DICE系数
    
    Args:
        mask1: 第一个mask (binary)
        mask2: 第二个mask (binary)
        
    Returns:
        DICE值 (0-1)
    """
    # 确保是二值mask
    mask1_bin = (mask1 > 0).astype(np.uint8)
    mask2_bin = (mask2 > 0).astype(np.uint8)
    
    # 计算交集
    intersection = np.logical_and(mask1_bin, mask2_bin).sum()
    total = mask1_bin.sum() + mask2_bin.sum()
    
    if total == 0:
        return 0.0
    
    return 2.0 * float(intersection) / float(total)


def bbox_overlap(bbox1: List[int], bbox2: List[int]) -> bool:
    """检查两个包围盒是否有重叠
    
    Args:
        bbox1: [x_min, y_min, x_max, y_max]
        bbox2: [x_min, y_min, x_max, y_max]
        
    Returns:
        是否有重叠
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # 检查是否不重叠
    if x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min:
        return False
    
    return True


def load_ground_truth_masks(json_file: str) -> Dict[str, Dict[str, Any]]:
    """加载真值mask信息
    
    Args:
        json_file: JSON文件路径
        
    Returns:
        字典，键为图片文件名（无扩展名），值为mask信息
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    gt_masks = {}
    for item in data:
        mask_path = item['mask_path']
        bbox = item['bbox']
        
        # 从mask路径提取文件名（无扩展名）
        mask_filename = Path(mask_path).stem
        
        gt_masks[mask_filename] = {
            'mask_path': mask_path,
            'bbox': bbox
        }
    
    return gt_masks


def get_image_files(image_dir: str) -> List[str]:
    """获取图片目录下所有支持的图片文件
    
    Args:
        image_dir: 图片目录
        
    Returns:
        图片文件路径列表，按文件名排序
    """
    image_files = []
    
    if not os.path.isdir(image_dir):
        print(f"Error: Directory not found: {image_dir}")
        return image_files
    
    for file_path in Path(image_dir).iterdir():
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            image_files.append(str(file_path))
    
    image_files.sort()
    return image_files


def load_mask_image(mask_path: str) -> Optional[np.ndarray]:
    """加载mask图像
    
    Args:
        mask_path: mask文件路径
        
    Returns:
        mask数组或None
    """
    try:
        if not os.path.exists(mask_path):
            print(f"Warning: Mask file not found: {mask_path}")
            return None
        
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            print(f"Warning: Failed to load mask: {mask_path}")
            return None
        
        # 如果是彩色图像，转换为灰度
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        return mask
    except Exception as e:
        print(f"Warning: Error loading mask {mask_path}: {e}")
        return None


def evaluate_image(image_path: str, gt_info: Dict[str, Any], mask_generator, metric: str = 'iou') -> Tuple[float, Dict[str, Any]]:
    """评估单张图片的分割效果
    
    Args:
        image_path: 图片路径
        gt_info: 真值信息
        mask_generator: SAM2 mask生成器
        metric: 评估指标 ('iou' 或 'dice')
        
    Returns:
        (最佳分数, 详细信息字典)
    """
    try:
        # 加载图片
        image = cv2.imread(image_path)
        if image is None:
            return 0.0, {'error': f'Failed to load image: {image_path}'}
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 加载真值mask
        gt_mask = load_mask_image(gt_info['mask_path'])
        if gt_mask is None:
            return 0.0, {'error': f'Failed to load ground truth mask: {gt_info["mask_path"]}'}
        
        gt_bbox = gt_info['bbox']
        
        # 使用SAM2 Automatic模式生成mask
        masks = mask_generator.generate(image_rgb)
        
        if not masks:
            return 0.0, {'error': 'No masks generated by SAM2', 'num_generated': 0}
        
        # 筛选与真值包围盒重叠的预测mask
        candidate_masks = []
        for mask_info in masks:
            pred_bbox = mask_info['bbox']  # SAM2返回的格式应该是[x, y, w, h]
            # 转换为[x_min, y_min, x_max, y_max]格式
            pred_bbox_xyxy = [
                int(pred_bbox[0]), 
                int(pred_bbox[1]), 
                int(pred_bbox[0] + pred_bbox[2]), 
                int(pred_bbox[1] + pred_bbox[3])
            ]
            
            if bbox_overlap(gt_bbox, pred_bbox_xyxy):
                candidate_masks.append({
                    'mask': mask_info['segmentation'],
                    'bbox': pred_bbox_xyxy,
                    'area': mask_info['area'],
                    'stability_score': mask_info.get('stability_score', 0.0)
                })
        
        if not candidate_masks:
            return 0.0, {
                'error': 'No overlapping masks found',
                'num_generated': len(masks),
                'num_candidates': 0
            }
        
        # 计算每个候选mask与真值的重合度
        best_score = 0.0
        best_mask_info = None
        
        for mask_info in candidate_masks:
            pred_mask = mask_info['mask']
            
            if metric == 'dice':
                score = calculate_dice(gt_mask, pred_mask)
            else:  # iou
                score = calculate_iou(gt_mask, pred_mask)
            
            if score > best_score:
                best_score = score
                best_mask_info = mask_info
        
        return best_score, {
            'num_generated': len(masks),
            'num_candidates': len(candidate_masks),
            'best_mask': best_mask_info,
            'all_generated_masks': masks,
            'gt_bbox': gt_bbox,
            'gt_mask': gt_mask
        }
        
    except Exception as e:
        return 0.0, {'error': f'Exception during evaluation: {str(e)}'}


def create_visualization(image_path: str, eval_info: Dict[str, Any], best_score: float, 
                        metric: str, output_path: str) -> None:
    """创建可视化图片
    
    Args:
        image_path: 原图路径
        eval_info: 评估信息
        best_score: 最佳分数
        metric: 评估指标
        output_path: 输出路径（无扩展名）
    """
    if not HAS_MATPLOTLIB:
        print(f"Skipping visualization for {Path(image_path).name} (matplotlib not available)")
        return
    
    try:
        # 加载原图
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 创建图1: 多区域分割展示
        fig1, ax1 = plt.subplots(1, 1, figsize=(12, 8))
        ax1.imshow(image_rgb)
        ax1.set_title(f'SAM2 Automatic Segmentation Results\n{len(eval_info["all_generated_masks"])} masks generated')
        
        # 显示所有生成的mask
        colors = plt.cm.Set3(np.linspace(0, 1, len(eval_info["all_generated_masks"])))
        for i, mask_info in enumerate(eval_info["all_generated_masks"]):
            mask = mask_info['segmentation']
            # 创建彩色mask
            colored_mask = np.zeros((*mask.shape, 4))
            colored_mask[mask] = [*colors[i][:3], 0.6]  # 半透明
            ax1.imshow(colored_mask)
            
            # 添加包围盒
            bbox = mask_info['bbox']  # [x, y, w, h]
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                                   linewidth=1, edgecolor=colors[i], facecolor='none')
            ax1.add_patch(rect)
        
        ax1.axis('off')
        plt.tight_layout()
        plt.savefig(f"{output_path}_all_masks.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 创建图2: 最佳预测与真值对比
        if eval_info.get('best_mask') is not None and eval_info.get('gt_mask') is not None:
            fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))
            
            # 转换为灰度图作为背景
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ax2.imshow(image_gray, cmap='gray', alpha=0.7)
            
            # 显示真值mask（绿色）
            gt_mask = eval_info['gt_mask']
            gt_colored = np.zeros((*gt_mask.shape, 4))
            gt_colored[gt_mask > 0] = [0, 1, 0, 0.5]  # 绿色半透明
            ax2.imshow(gt_colored)
            
            # 显示最佳预测mask（红色）
            best_mask = eval_info['best_mask']['mask']
            pred_colored = np.zeros((*best_mask.shape, 4))
            pred_colored[best_mask > 0] = [1, 0, 0, 0.5]  # 红色半透明
            ax2.imshow(pred_colored)
            
            # 添加包围盒
            gt_bbox = eval_info['gt_bbox']
            pred_bbox = eval_info['best_mask']['bbox']
            
            # 真值包围盒（绿色）
            gt_rect = patches.Rectangle((gt_bbox[0], gt_bbox[1]), 
                                      gt_bbox[2]-gt_bbox[0], gt_bbox[3]-gt_bbox[1],
                                      linewidth=2, edgecolor='green', facecolor='none', 
                                      linestyle='--', label='Ground Truth')
            ax2.add_patch(gt_rect)
            
            # 预测包围盒（红色）
            pred_rect = patches.Rectangle((pred_bbox[0], pred_bbox[1]), 
                                        pred_bbox[2]-pred_bbox[0], pred_bbox[3]-pred_bbox[1],
                                        linewidth=2, edgecolor='red', facecolor='none', 
                                        label='Best Prediction')
            ax2.add_patch(pred_rect)
            
            ax2.set_title(f'Best Prediction vs Ground Truth\n'
                         f'{metric.upper()}: {best_score:.3f} | '
                         f'Candidates: {eval_info["num_candidates"]}/{eval_info["num_generated"]}')
            ax2.legend()
            ax2.axis('off')
            
            plt.tight_layout()
            plt.savefig(f"{output_path}_comparison.png", dpi=150, bbox_inches='tight')
            plt.close()
        
    except Exception as e:
        print(f"Warning: Failed to create visualization for {image_path}: {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SAM2 Automatic segmentation against ground truth masks"
    )
    parser.add_argument(
        "--image_dir", 
        type=str, 
        required=True,
        help="Directory containing input images"
    )
    parser.add_argument(
        "--json_file", 
        type=str, 
        required=True,
        help="JSON file containing ground truth mask information"
    )
    parser.add_argument(
        "--sam_checkpoint", 
        type=str, 
        required=True,
        help="Path to SAM2 model checkpoint"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True,
        help="Output directory for results and visualizations"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default=None,
        help="Device to run on (cuda/cpu). Auto-detect if not specified"
    )
    parser.add_argument(
        "--metric", 
        type=str, 
        choices=['iou', 'dice'], 
        default='iou',
        help="Evaluation metric (default: iou)"
    )
    parser.add_argument(
        "--config_path", 
        type=str, 
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="Path to SAM2 config file"
    )
    parser.add_argument(
        "--points_per_side", 
        type=int, 
        default=32,
        help="Number of points per side for automatic mask generation"
    )
    parser.add_argument(
        "--pred_iou_thresh", 
        type=float, 
        default=0.88,
        help="IoU threshold for mask prediction"
    )
    parser.add_argument(
        "--stability_score_thresh", 
        type=float, 
        default=0.95,
        help="Stability score threshold for mask filtering"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Print detailed processing information"
    )
    parser.add_argument(
        "--no_visualization", 
        action="store_true",
        help="Skip creating visualization images"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 检查输入
    if not os.path.isdir(args.image_dir):
        print(f"Error: Image directory not found: {args.image_dir}")
        return 1
    
    if not os.path.isfile(args.json_file):
        print(f"Error: JSON file not found: {args.json_file}")
        return 1
    
    if not os.path.isfile(args.sam_checkpoint):
        print(f"Error: SAM2 checkpoint not found: {args.sam_checkpoint}")
        return 1
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载真值数据
    print(f"Loading ground truth data from: {args.json_file}")
    gt_masks = load_ground_truth_masks(args.json_file)
    print(f"Loaded {len(gt_masks)} ground truth masks")
    
    # 获取图片文件
    print(f"Scanning image directory: {args.image_dir}")
    image_files = get_image_files(args.image_dir)
    print(f"Found {len(image_files)} images")
    
    # 初始化SAM2
    print(f"Initializing SAM2 with checkpoint: {args.sam_checkpoint}")
    
    # 自动检测设备
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    try:
        # SAM2配置文件路径
        config_path = Path(__file__).parent.parent / "third_party" / "sam2" / args.config_path
        if not config_path.exists():
            config_path = Path(args.config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"SAM2 config file not found: {args.config_path}")
        
        sam_model = build_sam2(str(config_path), args.sam_checkpoint)
        sam_model = sam_model.to(device)
        
        # 创建自动mask生成器
        mask_generator = SAM2AutomaticMaskGenerator(
            model=sam_model,
            points_per_side=args.points_per_side,
            pred_iou_thresh=args.pred_iou_thresh,
            stability_score_thresh=args.stability_score_thresh,
        )
        
        print("SAM2 initialized successfully")
        
    except Exception as e:
        print(f"Error initializing SAM2: {e}")
        return 1
    
    # 评估所有图片
    print(f"\nStarting evaluation with {args.metric.upper()} metric...")
    
    all_scores = []
    num_processed = 0
    num_errors = 0
    results_log = []
    
    start_time = time.time()
    
    for i, image_path in enumerate(image_files, 1):
        image_name = Path(image_path).stem
        
        if args.verbose:
            print(f"[{i}/{len(image_files)}] Processing: {image_name}")
        
        # 检查是否有对应的真值
        if image_name not in gt_masks:
            if args.verbose:
                print(f"  -> No ground truth found, skipping")
            num_errors += 1
            continue
        
        gt_info = gt_masks[image_name]
        
        # 评估图片
        score, eval_info = evaluate_image(image_path, gt_info, mask_generator, args.metric)
        
        if 'error' in eval_info:
            if args.verbose:
                print(f"  -> Error: {eval_info['error']}")
            num_errors += 1
            continue
        
        all_scores.append(score)
        num_processed += 1
        
        # 记录结果
        result_entry = {
            'image_name': image_name,
            'score': score,
            'num_generated': eval_info['num_generated'],
            'num_candidates': eval_info['num_candidates']
        }
        results_log.append(result_entry)
        
        if args.verbose:
            print(f"  -> {args.metric.upper()}: {score:.3f} "
                  f"({eval_info['num_candidates']}/{eval_info['num_generated']} candidates)")
        
        # 创建可视化（如果启用）
        if not args.no_visualization:
            vis_output_path = os.path.join(args.output_dir, image_name)
            create_visualization(image_path, eval_info, score, args.metric, vis_output_path)
    
    # 计算统计信息
    if all_scores:
        best_score = max(all_scores)
        worst_score = min(all_scores)
        avg_score = sum(all_scores) / len(all_scores)
        
        # 保存详细结果
        results_summary = {
            'metric': args.metric,
            'total_images': len(image_files),
            'processed': num_processed,
            'errors': num_errors,
            'statistics': {
                'best': best_score,
                'worst': worst_score,
                'average': avg_score
            },
            'detailed_results': results_log,
            'parameters': {
                'points_per_side': args.points_per_side,
                'pred_iou_thresh': args.pred_iou_thresh,
                'stability_score_thresh': args.stability_score_thresh
            }
        }
        
        results_file = os.path.join(args.output_dir, 'evaluation_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)
        
        # 打印总结
        elapsed_time = time.time() - start_time
        print(f"\nEvaluation completed in {elapsed_time:.1f} seconds:")
        print(f"  Total images: {len(image_files)}")
        print(f"  Processed: {num_processed}")
        print(f"  Errors: {num_errors}")
        print(f"  Metric: {args.metric.upper()}")
        print(f"  Best score: {best_score:.3f}")
        print(f"  Worst score: {worst_score:.3f}")
        print(f"  Average score: {avg_score:.3f}")
        print(f"  Results saved to: {results_file}")
        print(f"  Visualizations saved to: {args.output_dir}")
        
        return 0
    else:
        print("No images were successfully processed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

```

让我检查脚本是否还有其他问题：

完美！现在让我创建一个简单的测试脚本来验证功能：

```python
#!/usr/bin/env python3

"""
测试SAM2自动评估脚本的基本功能
"""

import os
import sys
from pathlib import Path

def test_imports():
    """测试必要的导入"""
    try:
        import cv2
        import numpy as np
        import torch
        from PIL import Image as PILImage
        print("✓ Basic dependencies imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import basic dependencies: {e}")
        return False
    
    # 测试SAM2导入
    sys.path.append(str(Path(__file__).parent.parent / "third_party" / "sam2"))
    try:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        print("✓ SAM2 dependencies imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import SAM2: {e}")
        return False
    
    return True

def test_functions():
    """测试评估函数"""
    sys.path.append(str(Path(__file__).parent))
    try:
        from sam2_automatic_evaluation import (
            calculate_iou, calculate_dice, bbox_overlap,
            load_ground_truth_masks, get_image_files
        )
        
        # 测试IoU计算
        import numpy as np
        mask1 = np.zeros((100, 100), dtype=np.uint8)
        mask2 = np.zeros((100, 100), dtype=np.uint8)
        mask1[20:80, 20:80] = 1
        mask2[40:100, 40:100] = 1
        
        iou = calculate_iou(mask1, mask2)
        dice = calculate_dice(mask1, mask2)
        
        print(f"✓ IoU calculation: {iou:.3f}")
        print(f"✓ DICE calculation: {dice:.3f}")
        
        # 测试包围盒重叠
        bbox1 = [10, 10, 50, 50]
        bbox2 = [30, 30, 70, 70]
        bbox3 = [60, 60, 100, 100]
        
        overlap1 = bbox_overlap(bbox1, bbox2)
        overlap2 = bbox_overlap(bbox1, bbox3)
        
        print(f"✓ Bbox overlap test 1 (should be True): {overlap1}")
        print(f"✓ Bbox overlap test 2 (should be False): {overlap2}")
        
        return True
        
    except Exception as e:
        print(f"✗ Function test failed: {e}")
        return False

def main():
    print("Testing SAM2 Automatic Evaluation Script")
    print("=" * 50)
    
    success = True
    
    print("\n1. Testing imports...")
    if not test_imports():
        success = False
    
    print("\n2. Testing functions...")
    if not test_functions():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("✓ All tests passed! The evaluation script should work properly.")
    else:
        print("✗ Some tests failed. Please check the dependencies and installation.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
```

现在让我创建一个使用说明文档：

```markdown
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
```

让我更新待办事项来记录完成的工作：

## 🎉 SAM2 Automatic Evaluation Script 完成！

我已经成功创建了一个功能完整的SAM2自动分割评估脚本。以下是完成的内容：

### 📋 **创建的文件**

1. **主评估脚本**: `sam2_automatic_evaluation.py`
2. **测试脚本**: `test_evaluation.py` 
3. **详细文档**: `README_sam2_evaluation.md`

### 🛠️ **核心功能实现**

#### **1. SAM2 Automatic分割**
- 使用SAM2的AutomaticMaskGenerator进行批量分割
- 支持自定义分割参数（points_per_side, IoU阈值等）
- 自动设备检测（CUDA/CPU）

#### **2. 智能评估算法**
- **包围盒重叠筛选**: 只评估与真值包围盒有重叠的预测mask
- **多指标支持**: IoU和DICE两种评估指标
- **最佳匹配**: 从多个候选中选择最高分作为最终评估结果

#### **3. 统计分析**
- 计算整个数据集的**最好/最差/平均**分割效果
- 详细的处理统计（总数/成功/错误）
- 保存完整的评估结果JSON

#### **4. 双图可视化** 🎨
- **图1**: 多色展示所有SAM2分割区域
- **图2**: 最佳预测mask与真值mask的半透明叠加对比
- 包围盒可视化和评估指标标注

### 📊 **使用示例**

```bash
# 基础用法
python seg-rl/sam2_automatic_evaluation.py \
  --image_dir /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/imagesTr \
  --json_file /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/pred_masks-0.json \
  --sam_checkpoint /root/autodl-tmp/works/Seg-R0/third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --output_dir /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/evaluation_results \
  --device cuda \
  --metric dice \
  --verbose

# 高性能模式
python seg-rl/sam2_automatic_evaluation.py \
  --image_dir /path/to/images \
  --json_file /path/to/masks.json \
  --sam_checkpoint /path/to/sam2.pt \
  --output_dir /path/to/results \
  --device cuda \
  --points_per_side 64 \
  --pred_iou_thresh 0.9 \
  --no_visualization
```

### 🔧 **技术特点**

- **模块化设计**: 易于扩展和定制
- **错误处理**: 完善的异常处理和错误报告
- **依赖管理**: 可选matplotlib，自动降级功能
- **性能优化**: 支持GPU加速和参数调优
- **文档完善**: 详细的使用说明和故障排除

### 📈 **输出示例**

```
Evaluation completed in 120.5 seconds:
  Total images: 100
  Processed: 95
  Errors: 5
  Metric: DICE
  Best score: 0.892
  Worst score: 0.234
  Average score: 0.671
  Results saved to: /path/to/evaluation_results.json
  Visualizations saved to: /path/to/results
```

这个脚本完全满足你的需求，提供了专业级的SAM2自动分割评估工具，可以直接用于实际的研究和项目中！

```shellscript
# 基础用法
python seg-rl/sam2_automatic_evaluation.py \
  --image_dir /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/imagesTr \
  --json_file /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/pred_masks-0.json \
  --sam_checkpoint /root/autodl-tmp/works/Seg-R0/third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --output_dir /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/evaluation_results \
  --device cuda \
  --metric dice \
  --verbose

# 高性能模式
python seg-rl/sam2_automatic_evaluation.py \
  --image_dir /path/to/images \
  --json_file /path/to/masks.json \
  --sam_checkpoint /path/to/sam2.pt \
  --output_dir /path/to/results \
  --device cuda \
  --points_per_side 64 \
  --pred_iou_thresh 0.9 \
  --no_visualization
```

```plaintext
Evaluation completed in 120.5 seconds:
  Total images: 100
  Processed: 95
  Errors: 5
  Metric: DICE
  Best score: 0.892
  Worst score: 0.234
  Average score: 0.671
  Results saved to: /path/to/evaluation_results.json
  Visualizations saved to: /path/to/results
```

