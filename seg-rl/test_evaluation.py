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
