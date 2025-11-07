#!/usr/bin/env python3
"""
PEFT模块基础功能测试

测试各个模块的基本功能是否正常工作

调用示例:
    python -m seg-rl.peft.test_modules --device cpu
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

# 添加父目录到路径
_PARENT_DIR = Path(__file__).parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))


def test_lora_linear():
    """测试LoRA Linear层"""
    print("\n" + "="*80)
    print("Testing LoRALinear...")
    print("="*80)
    
    from peft.lora_sam2 import LoRALinear
    
    # 创建基础线性层
    base_linear = torch.nn.Linear(256, 256)
    
    # 创建LoRA层
    lora_layer = LoRALinear(base_linear, rank=16, alpha=32)
    
    # 测试前向传播
    x = torch.randn(2, 10, 256)
    y = lora_layer(x)
    
    assert y.shape == (2, 10, 256), f"Output shape mismatch: {y.shape}"
    assert not base_linear.weight.requires_grad, "Base linear should be frozen"
    assert lora_layer.lora_A.requires_grad, "LoRA A should be trainable"
    assert lora_layer.lora_B.requires_grad, "LoRA B should be trainable"
    
    print("✓ LoRALinear test passed")
    return True


def test_point_predictor():
    """测试点预测网络"""
    print("\n" + "="*80)
    print("Testing PointPredictorFromSAMFeatures...")
    print("="*80)
    
    from peft.point_predictor_peft import PointPredictorFromSAMFeatures
    
    # 创建网络
    predictor = PointPredictorFromSAMFeatures(
        sam_feature_dim=256,
        output_size=(512, 512),
        fusion_mode="film",
        feature_scale=8,
    )
    
    # 测试前向传播
    sam_features = torch.randn(2, 256, 64, 64)  # H/8, W/8
    prev_masks = torch.randn(2, 1, 512, 512)
    
    heatmap_logits, label_logits = predictor(sam_features, prev_masks)
    
    assert heatmap_logits.shape == (2, 1, 512, 512), f"Heatmap shape mismatch: {heatmap_logits.shape}"
    assert label_logits.shape == (2, 2), f"Label shape mismatch: {label_logits.shape}"
    
    print("✓ PointPredictorFromSAMFeatures test passed")
    return True


def test_utils():
    """测试工具函数"""
    print("\n" + "="*80)
    print("Testing utils...")
    print("="*80)
    
    from peft.utils_peft import compute_dice, compute_iou, compute_batch_pck
    
    # 测试Dice
    pred = torch.ones(100, 100)
    target = torch.ones(100, 100)
    dice = compute_dice(pred, target)
    assert abs(dice - 1.0) < 1e-6, f"Dice should be 1.0, got {dice}"
    
    # 测试IoU
    iou = compute_iou(pred, target)
    assert abs(iou - 1.0) < 1e-6, f"IoU should be 1.0, got {iou}"
    
    # 测试PCK
    pred_points = torch.tensor([[10.0, 10.0], [20.0, 20.0]])
    target_points = torch.tensor([[11.0, 11.0], [21.0, 21.0]])
    pck, distances = compute_batch_pck(pred_points, target_points, threshold=2.0)
    assert pck == 1.0, f"PCK should be 1.0, got {pck}"
    
    print("✓ Utils test passed")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("PEFT Modules Testing")
    print("="*80)
    print(f"Device: {args.device}")
    
    # 运行测试
    tests = [
        ("LoRALinear", test_lora_linear),
        ("PointPredictor", test_point_predictor),
        ("Utils", test_utils),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {name} test failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # 总结
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

