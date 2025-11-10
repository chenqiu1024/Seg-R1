#!/usr/bin/env python3
"""
查看实验配置的工具脚本

用法:
    # 查看训练配置
    python seg-rl/heatmap/show_experiment_config.py outputs/my_experiment/training_args.json
    
    # 查看推理配置
    python seg-rl/heatmap/show_experiment_config.py outputs/my_pred/inference_args.json
    
    # 对比两个实验
    python seg-rl/heatmap/show_experiment_config.py \
        outputs/exp1/training_args.json \
        outputs/exp2/training_args.json
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any


def format_value(value: Any) -> str:
    """格式化显示值"""
    if isinstance(value, bool):
        return "✓" if value else "✗"
    elif isinstance(value, (int, float)):
        if isinstance(value, float) and value < 1:
            return f"{value:.6f}"
        return str(value)
    elif isinstance(value, list):
        if len(value) <= 3:
            return str(value)
        return f"[{len(value)} items]"
    elif isinstance(value, str):
        if len(value) > 60:
            return value[:57] + "..."
        return value
    else:
        return str(value)


def show_config(path: str, title: str = None):
    """显示配置文件内容"""
    if not Path(path).exists():
        print(f"❌ 文件不存在: {path}")
        return None
    
    with open(path) as f:
        config = json.load(f)
    
    print(f"\n{'='*80}")
    if title:
        print(f"{title}")
    else:
        print(f"配置文件: {path}")
    print(f"{'='*80}")
    
    # 时间戳
    if 'timestamp' in config:
        print(f"\n📅 时间戳: {config['timestamp']}")
    
    # 关键参数
    print("\n🔑 关键参数:")
    key_params = [
        # 训练参数
        'use_sam_encoder', 'sam_lora_enabled', 'sam_lora_rank', 'sam_lora_alpha',
        'sam_lora_dropout', 'sam_lora_lr',
        'epochs', 'lr', 'batch_size', 'loss', 'arch',
        'height', 'width', 'sigma', 'tau',
        # 推理参数
        'model_path', 'num_points', 'device', 'resize',
    ]
    
    displayed = 0
    for k in key_params:
        if k in config:
            val = format_value(config[k])
            print(f"  {k:25s} = {val}")
            displayed += 1
    
    if displayed == 0:
        print("  (无关键参数)")
    
    # 数据路径
    print("\n📁 数据路径:")
    path_params = ['jsonl', 'sam_dir', 'images_dir', 'masks_dir', 'sam_masks_dir', 
                   'output_jsonl', 'sam_checkpoint', 'out_dir']
    displayed_paths = 0
    for k in path_params:
        if k in config:
            print(f"  {k:20s} = {config[k]}")
            displayed_paths += 1
    
    if displayed_paths == 0:
        print("  (无路径信息)")
    
    # 完整命令
    if 'command' in config:
        print(f"\n💻 完整命令:")
        cmd = config['command']
        # 格式化长命令（按空格分行）
        if len(cmd) > 100:
            parts = cmd.split()
            print(f"  {parts[0]}", end="")
            for i, part in enumerate(parts[1:], 1):
                if i % 3 == 0:
                    print(f" \\\n    {part}", end="")
                else:
                    print(f" {part}", end="")
            print()
        else:
            print(f"  {cmd}")
    
    print(f"{'='*80}\n")
    
    return config


def compare_configs(path1: str, path2: str):
    """对比两个配置文件的差异"""
    print(f"\n{'='*80}")
    print(f"配置对比")
    print(f"{'='*80}\n")
    
    config1 = show_config(path1, "实验 1")
    config2 = show_config(path2, "实验 2")
    
    if config1 is None or config2 is None:
        return
    
    # 找出差异
    all_keys = set(config1.keys()) | set(config2.keys())
    # 排除元数据字段
    exclude_keys = {'command', 'timestamp'}
    compare_keys = sorted(all_keys - exclude_keys)
    
    differences = []
    for key in compare_keys:
        val1 = config1.get(key)
        val2 = config2.get(key)
        if val1 != val2:
            differences.append((key, val1, val2))
    
    print(f"\n{'='*80}")
    print(f"差异对比 (共 {len(differences)} 处不同)")
    print(f"{'='*80}\n")
    
    if not differences:
        print("✅ 两个配置完全相同\n")
        return
    
    for key, val1, val2 in differences:
        print(f"📌 {key}:")
        print(f"  实验 1: {format_value(val1)}")
        print(f"  实验 2: {format_value(val2)}")
        print()


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print(__doc__)
        print("\n用法:")
        print("  python show_experiment_config.py <config.json>")
        print("  python show_experiment_config.py <config1.json> <config2.json>")
        sys.exit(1)
    
    if len(sys.argv) == 2:
        # 单个配置
        show_config(sys.argv[1])
    else:
        # 对比两个配置
        compare_configs(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()

