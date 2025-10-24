#!/usr/bin/env python3

"""
使用SAM2从点提示进行图像分割

读取由点提示组成的输入（支持JSON数组或JSONL逐行），使用SAM2进行分割，
将预测mask保存为灰度图像，并按要求生成/追加每幅图像对应的bbox日志。

功能:
- 从点提示生成图像分割mask
- 自动计算每个mask的最小包围盒
- 可选择输出包含mask路径和包围盒坐标的JSON文件

输入JSON格式（数组）或JSONL（逐行）:
    新格式: {"image": "/path/img.jpg", "points": [[x1,y1], [x2,y2]], "labels": [1, 0]}
    旧格式: {"image": "/path/img.jpg", "x": x1, "y": y1}

输出:
    - 预测mask按图像stem创建子目录保存为灰度图：output_dir/<stem>/<k>.png
      其中k为提示点序列长度-1；像素值：0=背景，255=前景
    - 在output_dir生成<stem>.jsonl（JSON对象，非数组）。该文件内容保持为：
      {"count": N, "bboxes": [[x1,y1,x2,y2], ...]}。每次新增一个mask，仅向
      bboxes追加一个新的bbox并将count递增，无需重写历史bbox。
    - 将输入JSON数组拷贝到 --json_output，并为每条加入/覆写
      "sam_masks_dir": output_dir

依赖项:
    - SAM2 (Segment Anything Model 2)
    - torch, PIL, opencv-python, numpy

安装SAM2:
    git clone https://github.com/facebookresearch/segment-anything-2.git third_party/sam2
    cd third_party/sam2 && pip install -e .

下载模型:
    # 下载SAM2.1 Hiera Large模型 (~900MB)
    wget -O third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
      https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

基础用法:
    python seg-rl/sam2_segment_from_points.py \
      --input_jsonl /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour.jsonl \
      --output_dir /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/pretrain_gt_masks \
      --sam_checkpoint /root/autodl-tmp/works/Seg-R0/third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
      --device cuda \

带JSON输出:
    python seg-rl/sam2_segment_from_points.py \
      --input_jsonl /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour.jsonl \
      --json_output /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour.jsonl \
      --output_dir /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/pretrain_gt_masks \
      --sam_checkpoint /root/autodl-tmp/works/Seg-R0/third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
      --device cuda \
      --resize 512 512 \
      --skip_existing

    python seg-rl/sam2_segment_from_points.py \
      --input_jsonl outputs/braintumour/pred_points-e160-251019.jsonl \
      --json_output outputs/braintumour/pred_points-e160-251019.jsonl \
      --output_dir outputs/braintumour/pred_masks-e160-251019 \
      --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
      --device cuda \
      --resize 512 512 \
      --skip_existing

    python seg-rl/sam2_segment_from_points.py \
      --input_jsonl outputs/braintumour/pred_points-sft_e185-251018.jsonl \
      --json_output outputs/braintumour/pred_points-sft_e185-251018.jsonl \
      --output_dir outputs/braintumour/pred_masks-sft_e185-251018 \
      --sam_checkpoint /root/autodl-tmp/works/Seg-R0/third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
      --device cuda \
      --resize 512 512 \
      --skip_existing
 
    /opt/anaconda3/envs/seg-r1/bin/python seg-rl/sam2_segment_from_points.py \
      --input_jsonl datasets/seg_r1_md/Task01_BrainTumour/pred_points-251001.jsonl \
      --json_output datasets/seg_r1_md/Task01_BrainTumour/pred_points-251001.jsonl \
      --output_dir outputs/braintumour/pretrain_pred_masks-251001 \
      --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
      --device mps \
      --resize 512 512 \
      --skip_existing

详细参数及输出格式说明请参考：docs/cursor_pretrain_flow_all_2025092801.md）：
    

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

# 导入热力图点预测模型与可视化工具（用于生成 heatmap-{i}.png）
try:
    from heatmap.model import ModelConfig as _HMModelConfig, PointHeatmapModel as _PointHeatmapModel  # type: ignore
    import torch.nn.functional as _F  # type: ignore
    from heatmap.utils import heatmap_to_pil as _heatmap_to_pil  # type: ignore
    _HEATMAP_AVAILABLE = True
except Exception:
    # 若用户未提供热力图权重或未安装依赖，则仅跳过热力图生成功能
    _HEATMAP_AVAILABLE = False

# 添加sam2路径
sys.path.append(str(Path(__file__).parent.parent / "third_party" / "sam2"))

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError as e:
    print(f"Error importing SAM2: {e}")
    print("Please ensure SAM2 is properly installed and the path is correct.")
    sys.exit(1)


def _print_progress(num_processed: int, num_skipped: int, num_errors: int, note: str) -> None:
    """在同一行输出进度信息"""
    msg = f"[Progress] processed={num_processed} skipped={num_skipped} errors={num_errors} | {note}"
    print(f"\r{msg}", end="", flush=True)


def to_abs(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    return os.path.abspath(path) if not os.path.isabs(path) else path


class SAMWrapper:
    """SAM2包装器，用于图像分割预测"""
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """初始化SAM2模型和预测器
        
        Args:
            model_path: SAM2模型检查点路径
            device: 运行设备 (e.g. "cuda", "cuda:0", "cpu")
                   如果为None，将自动检测可用设备
        """
        # 自动检测设备
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)
        
        # SAM2配置文件路径
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        sam_model = build_sam2(model_cfg, model_path, device=self.device) 
        # # 检查配置文件是否存在
        # config_path = Path(__file__).parent.parent / "third_party" / "sam2" / "sam2" / model_cfg
        # if not config_path.exists():
        #     # 尝试相对于当前目录
        #     config_path = Path(model_cfg)
        #     if not config_path.exists():
        #         raise FileNotFoundError(f"SAM2 config file not found: {model_cfg}")
        # sam_model = build_sam2(str(config_path), model_path)
        
        # # 移动到指定设备
        # sam_model = sam_model.to(self.device)
        
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
    """生成输出mask文件路径（旧：单文件路径；新：保留但未使用）
    
    Args:
        image_path: 输入图像路径
        output_dir: 输出目录
        
    Returns:
        输出mask文件路径
    """
    image_name = Path(image_path).stem
    return os.path.join(output_dir, f"{image_name}.png")


def get_mask_dir_for_image(image_path: str, output_dir: str) -> str:
    """返回该图像对应的mask子目录路径 output_dir/<stem>"""
    image_name = Path(image_path).stem
    return os.path.join(output_dir, image_name)


def get_mask_index_from_points(points: List[Tuple[float, float]]) -> int:
    """根据点序列长度返回mask索引：len(points)-1"""
    return max(0, int(len(points) - 1))


def get_mask_path_for_points(image_path: str, output_dir: str, points: List[Tuple[float, float]]) -> str:
    """输出mask路径：output_dir/<stem>/<k>.png，k=len(points)-1"""
    mask_dir = get_mask_dir_for_image(image_path, output_dir)
    os.makedirs(mask_dir, exist_ok=True)
    k = get_mask_index_from_points(points)
    return os.path.join(mask_dir, f"{k}.png")


def append_state_log(output_dir: str, image_path: str, bbox: Tuple[int, int, int, int]) -> None:
    """在 <output_dir>/<stem>.jsonl 中维护单个JSON对象：
    {"count": N, "bboxes": [[x1,y1,x2,y2], ...]}
    仅追加当前bbox并将count递增。不会重新读取或计算历史bbox。
    """
    stem = Path(image_path).stem
    log_path = os.path.join(output_dir, f"{stem}.jsonl")

    record: Dict[str, Any] = {"count": 0, "bboxes": []}
    if os.path.isfile(log_path):
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict) and "count" in parsed and "bboxes" in parsed:
                        record = parsed
        except Exception:
            pass

    # 追加当前bbox并递增count
    record.setdefault("bboxes", [])
    record["bboxes"].append([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])
    record["count"] = int(record.get("count", 0)) + 1

    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False)


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
                   help="Input JSON array (preferred) or JSONL file containing image paths and point coordinates")
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
                   help="Path to write the copied input JSON array with added 'sam_masks_dir'")
    # 额外：生成提示点概率热力图（heatmap-{i}.png）相关参数
    p.add_argument("--heatmap_model", type=str, default=None,
                   help="Path to heatmap predictor checkpoint (.pt); if set, also save heatmap-{i}.png")
    p.add_argument("--heatmap_tau", type=float, default=1.0,
                   help="Softmax temperature for probability heatmap generation")
    p.add_argument("--heatmap_size", type=int, nargs=2, default=None, metavar=("WIDTH", "HEIGHT"),
                   help="Inference size [width height] for heatmap model; default = use --resize or native size")
    return p.parse_args()


def main():
    args = parse_args()
    
    # 检查输入文件
    if not os.path.isfile(to_abs(args.input_jsonl)):
        print(f"Error: Input JSONL file not found: {args.input_jsonl}")
        return 1
    
    # 检查SAM2检查点
    if not os.path.isfile(to_abs(args.sam_checkpoint)):
        print(f"Error: SAM2 checkpoint not found: {args.sam_checkpoint}")
        return 1
    
    # 创建输出目录
    args.output_dir = to_abs(args.output_dir) or args.output_dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化SAM2
    print(f"Initializing SAM2 with checkpoint: {to_abs(args.sam_checkpoint)}")
    print(f"Using device: {args.device or 'auto-detect'}")
    
    try:
        sam_wrapper = SAMWrapper(to_abs(args.sam_checkpoint), args.device)
        print("SAM2 initialized successfully")
    except Exception as e:
        print(f"Error initializing SAM2: {e}")
        return 1
    
    # 可选：初始化热力图点预测模型
    hm_model = None
    hm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if (args.device is None) else torch.device(args.device)
    if _HEATMAP_AVAILABLE and args.heatmap_model:
        try:
            cfg = _HMModelConfig(backbone="unet_s", pretrained=False, main_in_channels=3, cond_in_channels=1)
            hm_model = _PointHeatmapModel(cfg).to(hm_device)
            ckpt = torch.load(to_abs(args.heatmap_model), map_location="cpu")
            sd = ckpt.get("model", ckpt)
            hm_model.load_state_dict(sd, strict=False)
            hm_model.eval()
            print(f"Heatmap model loaded: {to_abs(args.heatmap_model)}")
        except Exception as e:
            print(f"[WARN] Failed to load heatmap model: {e}. Skip heatmap generation.")
            hm_model = None

    # 读取输入（优先按JSON数组解析；失败则按JSONL逐行解析）
    print(f"Processing input: {to_abs(args.input_jsonl)}")
    
    num_processed = 0
    num_skipped = 0
    num_errors = 0
    
    # 解析输入记录列表 records: List[Dict]
    records: List[Dict[str, Any]] = []
    try:
        with open(to_abs(args.input_jsonl), 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content.startswith('['):
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    records = parsed
                else:
                    print("[ERROR] Input JSON root must be an array")
                    return 1
            else:
                # JSONL逐行
                for line_num, line in enumerate(content.splitlines(), 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"[ERROR] Line {line_num}: Invalid JSON - {e}")
                        num_errors += 1
                        continue
    except Exception as e:
        print(f"Error reading input: {e}")
        return 1

    for idx, obj in enumerate(records, 1):
        line_num = idx

        # 验证并提取点信息
        is_valid, points, labels = validate_and_extract_points(obj, line_num)
        if not is_valid:
            num_errors += 1
            continue

        image_path = to_abs(obj["image"]) or obj["image"]
        ### For Debug Only:
        # stem = Path(image_path).stem
        # if not stem == "BRATS_001_z0088":
        #     continue
        ### :For Debug Only

        # 检查图像文件是否存在
        if not os.path.isfile(image_path):
            print(f"[ERROR] Line {line_num}: Image file not found: {image_path}")
            num_errors += 1
            continue

        # 生成输出路径（子目录/<k>.png）
        output_path = get_mask_path_for_points(image_path, args.output_dir, points)

        # 检查是否跳过已存在的文件
        mask_existed = os.path.isfile(output_path)
        if args.skip_existing and mask_existed:
            _print_progress(num_processed, num_skipped + 1, num_errors,
                            f"skip line {line_num}: {Path(output_path).name} exists")
            # 此分支无现成bbox，且不应为不存在的日志补写空记录
            num_skipped += 1
            continue

        try:
            # 加载图像
            image = PILImage.open(to_abs(image_path)).convert("RGB")
            orig_w, orig_h = image.size

            # 若指定resize，则按比例缩放点坐标并对图像进行resize
            if args.resize:
                resize_w, resize_h = int(args.resize[0]), int(args.resize[1])
                scale_x = float(resize_w) / float(orig_w)
                scale_y = float(resize_h) / float(orig_h)
                points_resized = [(px * scale_x, py * scale_y) for (px, py) in points]
                image_for_pred = image.resize((resize_w, resize_h), PILImage.BILINEAR)
            else:
                points_resized = points
                image_for_pred = image
                scale_x = 1.0
                scale_y = 1.0

            # 运行SAM2预测（在处理后的分辨率上）
            mask, confidence = sam_wrapper.predict(image_for_pred, points_resized, labels)

            # 如果做了resize，则将mask缩放回原图尺寸，并在原图尺寸上计算bbox
            if args.resize:
                mask_binary = (mask > 0).astype(np.uint8)
                mask_orig_size = cv2.resize(mask_binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                x_min, y_min, x_max, y_max = calculate_bounding_box(mask_orig_size)
                # 保存缩放回原尺寸的mask
                save_mask_as_grayscale(mask_orig_size, output_path)
            else:
                # 未resize，直接使用原mask与bbox
                x_min, y_min, x_max, y_max = calculate_bounding_box(mask)
                save_mask_as_grayscale(mask, output_path)

            # 追加/更新该图像的日志（单一JSON对象，追加bbox并递增count）
            try:
                append_state_log(args.output_dir, image_path, (x_min, y_min, x_max, y_max))
            except Exception as e:
                print(f"[WARN] Failed to append state log for {image_path}: {e}")

            # 可选：生成并保存提示点概率热力图 heatmap-{k}.png
            # 语义：对于第 i 步（k=i），热力图来源于“图像 + 上一步的预测掩膜（i-1步；若i=0则为全零掩膜）”。
            if hm_model is not None:
                try:
                    # 选择热力图推理尺寸（优先 heatmap_size，其次使用 --resize，最后使用原始尺寸）
                    if args.heatmap_size is not None:
                        hm_w, hm_h = int(args.heatmap_size[0]), int(args.heatmap_size[1])
                    elif args.resize is not None:
                        hm_w, hm_h = int(args.resize[0]), int(args.resize[1])
                    else:
                        hm_w, hm_h = orig_w, orig_h

                    # 构造“上一时刻”的灰度掩膜（若 i>0 且存在文件，则直接读；否则为了稳健性直接用上一时刻的SAM预测）
                    k = get_mask_index_from_points(points)
                    prev_gray_pil: Optional[PILImage.Image]
                    if k <= 0:
                        prev_gray_pil = PILImage.new("L", (hm_w, hm_h), 0)
                    else:
                        mask_dir = get_mask_dir_for_image(image_path, args.output_dir)
                        prev_path = os.path.join(mask_dir, f"{k-1}.png")
                        if os.path.isfile(prev_path):
                            prev_gray_pil = PILImage.open(prev_path).convert("L").resize((hm_w, hm_h), PILImage.NEAREST)
                        else:
                            # 若上一步掩膜文件缺失，则用上一时刻点再次运行一次SAM以得到上一时刻掩膜
                            prev_pts = points_resized[:-1]
                            prev_labs = labels[:-1] if labels else None
                            prev_mask_np, _ = sam_wrapper.predict(
                                image_for_pred.resize((hm_w, hm_h), PILImage.BILINEAR), prev_pts, prev_labs
                            )
                            prev_gray = (prev_mask_np.astype(np.uint8) * 255)
                            prev_gray_pil = PILImage.fromarray(prev_gray, mode="L")

                    # 构造模型输入张量：RGB 与 灰度条件（范围与归一化与训练一致）
                    # 注意：这里独立于 SAM2 的 resize，确保热力图与指定 hm_w/h 对齐
                    from torchvision.transforms import functional as TF  # type: ignore
                    rgb_for_hm = image.resize((hm_w, hm_h), PILImage.BILINEAR)
                    rgb_t = TF.to_tensor(rgb_for_hm)
                    rgb_t = TF.normalize(rgb_t, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)).unsqueeze(0)
                    gray_pil = prev_gray_pil if prev_gray_pil is not None else PILImage.new("L", (hm_w, hm_h), 0)
                    g_t = TF.to_tensor(gray_pil)
                    g_t = ((g_t - 0.5) / 0.5).unsqueeze(0)

                    with torch.no_grad():
                        logits, _ = hm_model(rgb_t.to(hm_device), g_t.to(hm_device))  # [1,1,H,W]
                        # 将 logits 转为概率热力图（按像素 softmax），与训练/可视化一致
                        b, c, Hh, Wh = logits.shape
                        p = _F.softmax(logits.view(b, -1) / max(float(args.heatmap_tau), 1e-6), dim=1)
                        prob = p.view(Hh, Wh).detach().cpu().float().numpy()

                    # 映射为彩色图并按原图尺寸保存，便于与 {k}.png 对齐查看
                    hm_img = _heatmap_to_pil(prob)
                    if (hm_w, hm_h) != (orig_w, orig_h):
                        hm_img = hm_img.resize((orig_w, orig_h), PILImage.BILINEAR)
                    heatmap_path = os.path.join(get_mask_dir_for_image(image_path, args.output_dir), f"heatmap-{k}.png")
                    os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
                    hm_img.save(heatmap_path)
                except Exception as e:
                    print(f"[WARN] Failed to save heatmap for {image_path}: {e}")

            _print_progress(num_processed + 1, num_skipped, num_errors,
                            f"ok line {line_num}: {Path(output_path).name} conf={confidence:.3f}")
            num_processed += 1

        except Exception as e:
            _print_progress(num_processed, num_skipped, num_errors + 1,
                            f"error line {line_num}: {Path(image_path).name}")
            num_errors += 1
            continue
    
    # 将输入数组复制到json_output并添加/覆盖 sam_masks_dir
    if args.json_output:
        try:
            # 若前面用JSONL解析得到records，则以records为准写出数组
            out_records = []
            for obj in records:
                if isinstance(obj, dict):
                    obj2 = dict(obj)
                    obj2["sam_masks_dir"] = args.output_dir
                    out_records.append(obj2)
            with open(args.json_output, 'w', encoding='utf-8') as jf:
                json.dump(out_records, jf, indent=2, ensure_ascii=False)
            print(f"Copied input to json_output with 'sam_masks_dir': {args.json_output}")
        except Exception as e:
            print(f"[ERROR] Failed to save json_output: {e}")
            return 1
    
    # 打印总结（先补换行清空进度行）
    print()
    print(f"Processing completed:")
    print(f"  Processed: {num_processed}")
    print(f"  Skipped: {num_skipped}")
    print(f"  Errors: {num_errors}")
    print(f"  Output directory: {args.output_dir}")
    if args.json_output:
        print(f"  JSON output: {args.json_output}")
    
    return 0 if num_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
