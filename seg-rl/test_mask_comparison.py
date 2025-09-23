#!/usr/bin/env python3
"""
Test script for mask_comparison.py

This script creates sample data and tests the mask comparison functionality.
"""

import os
import numpy as np
import cv2
from pathlib import Path
import tempfile
import subprocess
import sys

def create_test_data(temp_dir):
    """Create sample test data"""
    # Create directories
    pred_dir = os.path.join(temp_dir, "predictions")
    gt_dir = os.path.join(temp_dir, "ground_truth")
    img_dir = os.path.join(temp_dir, "images")
    
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    
    # Create sample masks and images
    for i in range(5):
        # Create ground truth mask (circle)
        gt_mask = np.zeros((256, 256), dtype=np.uint8)
        center = (128, 128)
        radius = 50 + i * 10
        cv2.circle(gt_mask, center, radius, 255, -1)
        
        # Create prediction mask (slightly offset circle)
        pred_mask = np.zeros((256, 256), dtype=np.uint8)
        pred_center = (128 + i * 5, 128 + i * 3)  # Slight offset
        pred_radius = radius + i * 2  # Slightly different size
        cv2.circle(pred_mask, pred_center, pred_radius, 255, -1)
        
        # Create background image (random noise)
        bg_img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        
        # Save files
        filename = f"test_{i:03d}"
        cv2.imwrite(os.path.join(gt_dir, f"{filename}.png"), gt_mask)
        cv2.imwrite(os.path.join(pred_dir, f"{filename}.png"), pred_mask)
        cv2.imwrite(os.path.join(img_dir, f"{filename}.jpg"), bg_img)
    
    return pred_dir, gt_dir, img_dir

def test_mask_comparison():
    """Test the mask comparison script"""
    print("Creating test data...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        pred_dir, gt_dir, img_dir = create_test_data(temp_dir)
        output_dir = os.path.join(temp_dir, "output")
        
        # Test basic functionality
        print("Testing basic mask comparison...")
        cmd = [
            sys.executable, "mask_comparison.py",
            "--pred_dir", pred_dir,
            "--gt_dir", gt_dir,
            "--img_dir", img_dir,
            "--output_dir", output_dir,
            "--summary_output", os.path.join(temp_dir, "summary.png"),
            "--results_json", os.path.join(temp_dir, "results.json"),
            "--visualize_all",
            "--summary_samples", "3"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("✓ Basic test passed")
            print("Standard output:")
            print(result.stdout)
            
            # Check if output files were created
            summary_path = os.path.join(temp_dir, "summary.png")
            results_path = os.path.join(temp_dir, "results.json")
            
            if os.path.exists(summary_path):
                print("✓ Summary visualization created")
            else:
                print("✗ Summary visualization not found")
                
            if os.path.exists(results_path):
                print("✓ Results JSON created")
            else:
                print("✗ Results JSON not found")
                
            # Check individual visualizations
            if os.path.exists(output_dir):
                individual_files = os.listdir(output_dir)
                print(f"✓ {len(individual_files)} individual comparison images created")
            else:
                print("✗ Output directory not created")
                
        except subprocess.CalledProcessError as e:
            print("✗ Test failed with error:")
            print("Standard output:", e.stdout)
            print("Standard error:", e.stderr)
            return False
        
        # Test without images
        print("\nTesting without background images...")
        cmd_no_img = [
            sys.executable, "mask_comparison.py",
            "--pred_dir", pred_dir,
            "--gt_dir", gt_dir,
            "--summary_output", os.path.join(temp_dir, "summary_no_img.png"),
            "--no_summary" # Test this flag too
        ]
        
        try:
            result = subprocess.run(cmd_no_img, capture_output=True, text=True, check=True)
            print("✓ Test without background images passed")
        except subprocess.CalledProcessError as e:
            print("✗ Test without background images failed:")
            print("Standard error:", e.stderr)
            return False
    
    print("\n✓ All tests passed!")
    return True

if __name__ == "__main__":
    print("Testing mask comparison script...")
    test_mask_comparison()
