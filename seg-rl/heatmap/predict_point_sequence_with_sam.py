#!/usr/bin/env python3

"""
整合脚本：自动执行模型预测点序列并配合SAM2分割的完整流程

该脚本整合了以下两个步骤的交替执行过程：
1. 使用监督训练得到的模型预测下一个提示点（predict_next_point_from_model.py）
2. 使用SAM2生成掩模（sam2_segment_from_points.py）

使用示例：

# 基础用法：生成8个点的序列
python seg-rl/heatmap/predict_point_sequence_with_sam.py \
  --model_path outputs/braintumour/points_predictor-251001-160epochs.pt \
  --images_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/images \
  --masks_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/masks \
  --output_jsonl outputs/braintumour/pred_points-e160-251019.jsonl \
  --sam_masks_dir outputs/braintumour/pred_masks-e160-251019 \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --device cuda \
  --resize 512 512 \
  --num_points 8

# 完整参数示例
python seg-rl/heatmap/predict_point_sequence_with_sam.py \
  --model_path outputs/braintumour/points_predictor-251001-160epochs.pt \
  --images_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/images \
  --masks_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/masks \
  --output_jsonl outputs/braintumour/pred_points-e160-251019.jsonl \
  --sam_masks_dir outputs/braintumour/pred_masks-e160-251019 \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --device cuda \
  --resize 512 512 \
  --num_points 16 \
  --skip_existing \
  --height 512 \
  --width 512 \
  --tau 1.0 \
  --amp

# 使用自定义脚本路径
python seg-rl/heatmap/predict_point_sequence_with_sam.py \
  --model_path outputs/braintumour/points_predictor-251001-160epochs.pt \
  --images_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/images \
  --masks_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/masks \
  --output_jsonl outputs/braintumour/pred_points-e160-251019.jsonl \
  --sam_masks_dir outputs/braintumour/pred_masks-e160-251019 \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --device cuda \
  --resize 512 512 \
  --num_points 8 \
  --predict_script seg-rl/heatmap/predict_next_point_from_model.py \
  --sam_segment_script seg-rl/sam2_segment_from_points.py

工作流程：
1. 首次使用模型预测第一个提示点（基于RGB图像和零掩模）
2. 使用SAM2生成第一个掩模
3. 循环执行（num_points-1次）：
   a. 使用模型基于RGB图像和最新SAM2掩模预测下一个提示点
   b. 使用SAM2生成新的掩模
4. 最终输出包含完整点序列和所有掩模的JSONL文件

输出格式：
- output_jsonl: JSON数组，每个元素包含：
  {
    "image": "/abs/path/to/image.jpg",
    "gt_mask": "/abs/path/to/mask.png",
    "points": [[x0,y0], [x1,y1], ..., [xN,yN]],
    "labels": [l0, l1, ..., lN],
    "sam_masks_dir": "/path/to/sam_masks_dir"
  }
- sam_masks_dir/<stem>/0.png, 1.png, ..., (N-1).png: 各步骤生成的掩模
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


def to_abs(path: Optional[str]) -> Optional[str]:
    """将路径转换为绝对路径"""
    if path is None:
        return None
    return os.path.abspath(path) if not os.path.isabs(path) else path


def run_command(cmd: list, description: str) -> bool:
    """执行命令并返回是否成功"""
    print(f"\n{'='*60}")
    print(f"[执行] {description}")
    print(f"[命令] {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        print(f"[成功] {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[错误] {description} 失败: {e}")
        return False
    except Exception as e:
        print(f"[错误] 执行命令时发生异常: {e}")
        return False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="整合脚本：自动执行模型预测点序列并配合SAM2分割的完整流程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # 必需参数
    p.add_argument("--model_path", type=str, required=True,
                   help="训练好的模型检查点路径（.pt文件）")
    p.add_argument("--images_dir", type=str, required=True,
                   help="图像目录路径（包含JPG/PNG等图像文件）")
    p.add_argument("--masks_dir", type=str, required=True,
                   help="掩模目录路径（包含PNG掩模文件，与图像对齐，用于填充gt_mask字段）")
    p.add_argument("--output_jsonl", type=str, required=True,
                   help="输出JSONL文件路径（JSON数组格式）")
    p.add_argument("--sam_masks_dir", type=str, required=True,
                   help="SAM2掩模输出目录")
    p.add_argument("--sam_checkpoint", type=str, required=True,
                   help="SAM2模型检查点路径")
    p.add_argument("--num_points", type=int, required=True,
                   help="要生成的点序列长度（至少为1）")
    
    # SAM2相关参数
    p.add_argument("--device", type=str, default="cuda",
                   help="运行设备（cuda/cpu/mps等，默认: cuda）")
    p.add_argument("--resize", type=int, nargs=2, default=None, metavar=("WIDTH", "HEIGHT"),
                   help="图像resize尺寸 [width height]（例如: 512 512）")
    p.add_argument("--skip_existing", action="store_true",
                   help="如果输出文件已存在则跳过处理")
    
    # 模型预测相关参数
    p.add_argument("--height", type=int, default=0,
                   help="模型推理时的图像高度（0=使用原始尺寸，默认: 0）")
    p.add_argument("--width", type=int, default=0,
                   help="模型推理时的图像宽度（0=使用原始尺寸，默认: 0）")
    p.add_argument("--tau", type=float, default=1.0,
                   help="Softmax温度参数，用于热图到概率的转换（默认: 1.0）")
    p.add_argument("--amp", action="store_true",
                   help="启用自动混合精度（仅CUDA设备）")
    
    # 脚本路径（可选，用于指定自定义脚本位置）
    p.add_argument("--predict_script", type=str,
                   default="seg-rl/heatmap/predict_next_point_from_model.py",
                   help="点预测脚本路径（默认: seg-rl/heatmap/predict_next_point_from_model.py）")
    p.add_argument("--sam_segment_script", type=str,
                   default="seg-rl/sam2_segment_from_points.py",
                   help="SAM2分割脚本路径（默认: seg-rl/sam2_segment_from_points.py）")
    
    return p.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[bool, str]:
    """验证参数有效性"""
    if args.num_points < 1:
        return False, "num_points 必须 >= 1"
    
    if not os.path.isdir(to_abs(args.images_dir) or args.images_dir):
        return False, f"images_dir 不存在: {args.images_dir}"
    
    if not os.path.isdir(to_abs(args.masks_dir) or args.masks_dir):
        return False, f"masks_dir 不存在: {args.masks_dir}"
    
    if not os.path.isfile(to_abs(args.model_path) or args.model_path):
        return False, f"model_path 不存在: {args.model_path}"
    
    if not os.path.isfile(to_abs(args.sam_checkpoint) or args.sam_checkpoint):
        return False, f"sam_checkpoint 不存在: {args.sam_checkpoint}"
    
    predict_script = to_abs(args.predict_script) or args.predict_script
    if not os.path.isfile(predict_script):
        return False, f"predict_script 不存在: {predict_script}"
    
    sam_script = to_abs(args.sam_segment_script) or args.sam_segment_script
    if not os.path.isfile(sam_script):
        return False, f"sam_segment_script 不存在: {sam_script}"
    
    return True, ""


def build_predict_cmd(args: argparse.Namespace, is_first: bool) -> list:
    """构建点预测命令"""
    script_path = to_abs(args.predict_script) or args.predict_script
    # 使用模块方式调用（因为脚本在包内）
    cmd = [sys.executable, "-m", "seg-rl.heatmap.predict_next_point_from_model"]
    
    if is_first:
        # 首次生成模式
        cmd.extend([
            "--model_path", args.model_path,
            "--images_dir", args.images_dir,
            "--masks_dir", args.masks_dir,
            "--output_json", args.output_jsonl,
        ])
        if args.sam_masks_dir:
            cmd.extend(["--sam_dir", args.sam_masks_dir])
    else:
        # 追加模式
        cmd.extend([
            "--model_path", args.model_path,
            "--appendto_json", args.output_jsonl,
            "--sam_dir", args.sam_masks_dir,
        ])
    
    # 模型推理参数
    if args.height > 0:
        cmd.extend(["--height", str(args.height)])
    if args.width > 0:
        cmd.extend(["--width", str(args.width)])
    if args.tau != 1.0:
        cmd.extend(["--tau", str(args.tau)])
    if args.amp:
        cmd.append("--amp")
    
    # SAM checkpoint（如果模型使用 SAM encoder）
    if args.sam_checkpoint:
        cmd.extend(["--sam_checkpoint", args.sam_checkpoint])
    
    return cmd


def build_sam_segment_cmd(args: argparse.Namespace) -> list:
    """构建SAM2分割命令"""
    script_path = to_abs(args.sam_segment_script) or args.sam_segment_script
    cmd = [sys.executable, script_path]
    
    cmd.extend([
        "--input_jsonl", args.output_jsonl,
        "--json_output", args.output_jsonl,
        "--output_dir", args.sam_masks_dir,
        "--sam_checkpoint", args.sam_checkpoint,
        "--device", args.device,
    ])
    
    if args.resize:
        cmd.extend(["--resize", str(args.resize[0]), str(args.resize[1])])
    
    if args.skip_existing:
        cmd.append("--skip_existing")
    
    return cmd


def main() -> int:
    args = parse_args()
    
    # 验证参数
    is_valid, error_msg = validate_args(args)
    if not is_valid:
        print(f"[错误] 参数验证失败: {error_msg}")
        return 1
    
    # 确保输出目录存在
    output_jsonl_dir = os.path.dirname(args.output_jsonl) or "."
    os.makedirs(output_jsonl_dir, exist_ok=True)
    os.makedirs(args.sam_masks_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("开始执行模型预测点序列生成流程")
    print(f"{'='*60}")
    print(f"模型路径: {args.model_path}")
    print(f"图像目录: {args.images_dir}")
    print(f"掩模目录: {args.masks_dir}")
    print(f"输出JSONL: {args.output_jsonl}")
    print(f"SAM掩模目录: {args.sam_masks_dir}")
    print(f"点序列长度: {args.num_points}")
    print(f"设备: {args.device}")
    if args.resize:
        print(f"Resize: {args.resize[0]}x{args.resize[1]}")
    if args.height > 0 or args.width > 0:
        print(f"模型推理尺寸: {args.width}x{args.height}")
    print(f"{'='*60}\n")
    
    # 步骤1: 使用模型预测第一个提示点
    print(f"\n[步骤 1/{args.num_points*2-1}] 使用模型预测第一个提示点")
    cmd_predict_first = build_predict_cmd(args, is_first=True)
    if not run_command(cmd_predict_first, "使用模型预测第一个提示点"):
        print("[错误] 预测第一个提示点失败")
        return 1
    
    # 步骤2: 使用SAM2生成第一个掩模
    print(f"\n[步骤 2/{args.num_points*2-1}] 使用SAM2生成第一个掩模")
    cmd_sam_first = build_sam_segment_cmd(args)
    if not run_command(cmd_sam_first, "使用SAM2生成第一个掩模"):
        print("[错误] 生成第一个掩模失败")
        return 1
    
    # 步骤3-N: 循环预测后续点并生成掩模
    for i in range(1, args.num_points):
        step_num = i * 2 + 1
        total_steps = args.num_points * 2 - 1
        
        # 使用模型预测第(i+1)个提示点
        print(f"\n[步骤 {step_num}/{total_steps}] 使用模型预测第 {i+1} 个提示点")
        cmd_predict_next = build_predict_cmd(args, is_first=False)
        if not run_command(cmd_predict_next, f"使用模型预测第 {i+1} 个提示点"):
            print(f"[错误] 预测第 {i+1} 个提示点失败")
            return 1
        
        # 使用SAM2生成第(i+1)个掩模
        step_num += 1
        print(f"\n[步骤 {step_num}/{total_steps}] 使用SAM2生成第 {i+1} 个掩模")
        cmd_sam_next = build_sam_segment_cmd(args)
        if not run_command(cmd_sam_next, f"使用SAM2生成第 {i+1} 个掩模"):
            print(f"[错误] 生成第 {i+1} 个掩模失败")
            return 1
    
    # 完成
    print(f"\n{'='*60}")
    print("模型预测点序列生成流程完成！")
    print(f"{'='*60}")
    print(f"输出JSONL: {args.output_jsonl}")
    print(f"SAM掩模目录: {args.sam_masks_dir}")
    print(f"共生成 {args.num_points} 个点的序列")
    print(f"{'='*60}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

