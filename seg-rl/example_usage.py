#!/usr/bin/env python3
"""
Example usage of the mask comparison tool

This script demonstrates how to use the mask_comparison.py script programmatically
and shows different usage patterns.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_mask_comparison(pred_dir, gt_dir, img_dir=None, output_dir=None, **kwargs):
    """
    Run mask comparison with specified parameters
    
    Args:
        pred_dir: Directory with predicted masks
        gt_dir: Directory with ground truth masks
        img_dir: Directory with original images (optional)
        output_dir: Output directory for individual comparisons (optional)
        **kwargs: Additional arguments for the script
    
    Returns:
        subprocess.CompletedProcess: The result of running the script
    """
    
    cmd = [sys.executable, "mask_comparison.py", "--pred_dir", pred_dir, "--gt_dir", gt_dir]
    
    if img_dir:
        cmd.extend(["--img_dir", img_dir])
    
    if output_dir:
        cmd.extend(["--output_dir", output_dir])
    
    # Add additional arguments
    for key, value in kwargs.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])
    
    return subprocess.run(cmd, capture_output=True, text=True)

def example_1_basic_evaluation():
    """Example 1: Basic mask evaluation without visualization"""
    print("Example 1: Basic Evaluation")
    print("-" * 40)
    
    # Assume you have these directories
    pred_dir = "path/to/predictions"
    gt_dir = "path/to/ground_truth"
    
    # Just run evaluation without any visualization
    result = run_mask_comparison(
        pred_dir=pred_dir,
        gt_dir=gt_dir,
        no_summary=True
    )
    
    if result.returncode == 0:
        print("Evaluation completed successfully!")
        print(result.stdout)
    else:
        print("Error occurred:")
        print(result.stderr)

def example_2_full_analysis():
    """Example 2: Complete analysis with all features"""
    print("Example 2: Complete Analysis")
    print("-" * 40)
    
    # Input directories
    pred_dir = "path/to/predictions"
    gt_dir = "path/to/ground_truth"
    img_dir = "path/to/original_images"
    
    # Output locations
    output_dir = "results/individual_comparisons"
    summary_output = "results/summary.png"
    results_json = "results/evaluation_metrics.json"
    
    # Create output directory
    os.makedirs("results", exist_ok=True)
    
    result = run_mask_comparison(
        pred_dir=pred_dir,
        gt_dir=gt_dir,
        img_dir=img_dir,
        output_dir=output_dir,
        summary_output=summary_output,
        results_json=results_json,
        summary_samples=8,
        visualize_all=True,
        random_seed=42
    )
    
    if result.returncode == 0:
        print("Complete analysis finished!")
        print(f"Individual comparisons saved to: {output_dir}")
        print(f"Summary visualization saved to: {summary_output}")
        print(f"Detailed metrics saved to: {results_json}")
    else:
        print("Error occurred:")
        print(result.stderr)

def example_3_custom_summary():
    """Example 3: Custom summary visualization only"""
    print("Example 3: Custom Summary")
    print("-" * 40)
    
    pred_dir = "path/to/predictions"
    gt_dir = "path/to/ground_truth"
    
    result = run_mask_comparison(
        pred_dir=pred_dir,
        gt_dir=gt_dir,
        summary_output="custom_summary_12_samples.png",
        summary_samples=12,
        random_seed=123
    )
    
    if result.returncode == 0:
        print("Custom summary created!")
    else:
        print("Error occurred:")
        print(result.stderr)

def example_4_batch_processing():
    """Example 4: Process multiple datasets in batch"""
    print("Example 4: Batch Processing")
    print("-" * 40)
    
    datasets = [
        ("dataset1/predictions", "dataset1/ground_truth", "results/dataset1"),
        ("dataset2/predictions", "dataset2/ground_truth", "results/dataset2"),
        ("dataset3/predictions", "dataset3/ground_truth", "results/dataset3"),
    ]
    
    overall_results = []
    
    for pred_dir, gt_dir, result_dir in datasets:
        print(f"Processing {pred_dir}...")
        
        # Create result directory
        os.makedirs(result_dir, exist_ok=True)
        
        # Run evaluation
        result = run_mask_comparison(
            pred_dir=pred_dir,
            gt_dir=gt_dir,
            summary_output=f"{result_dir}/summary.png",
            results_json=f"{result_dir}/metrics.json",
            summary_samples=6
        )
        
        if result.returncode == 0:
            print(f"✓ {pred_dir} completed")
            overall_results.append((pred_dir, "success"))
        else:
            print(f"✗ {pred_dir} failed: {result.stderr}")
            overall_results.append((pred_dir, "failed"))
    
    # Summary of batch processing
    print("\nBatch Processing Summary:")
    for dataset, status in overall_results:
        print(f"  {dataset}: {status}")

def show_help():
    """Show available command line options"""
    print("Available Command Line Options:")
    print("-" * 40)
    
    result = subprocess.run([sys.executable, "mask_comparison.py", "--help"], 
                          capture_output=True, text=True)
    print(result.stdout)

if __name__ == "__main__":
    print("Mask Comparison Tool - Usage Examples")
    print("=" * 50)
    
    # Show help first
    show_help()
    
    print("\n" + "=" * 50)
    print("USAGE EXAMPLES")
    print("=" * 50)
    
    # Note: These examples won't actually run since the paths don't exist
    # They are meant to show the patterns of usage
    
    print("\n# Example 1: Basic evaluation")
    print("python mask_comparison.py --pred_dir predictions/ --gt_dir ground_truth/ --no_summary")
    
    print("\n# Example 2: Full analysis with original images")
    print("python mask_comparison.py \\")
    print("    --pred_dir predictions/ \\")
    print("    --gt_dir ground_truth/ \\")
    print("    --img_dir original_images/ \\")
    print("    --output_dir results/individual/ \\")
    print("    --summary_output results/summary.png \\")
    print("    --results_json results/metrics.json \\")
    print("    --visualize_all \\")
    print("    --summary_samples 8")
    
    print("\n# Example 3: Quick summary with custom sampling")
    print("python mask_comparison.py \\")
    print("    --pred_dir predictions/ \\")
    print("    --gt_dir ground_truth/ \\")
    print("    --summary_output quick_summary.png \\")
    print("    --summary_samples 12 \\")
    print("    --random_seed 42")
    
    print("\n# Example 4: Evaluation only (no visualization)")
    print("python mask_comparison.py \\")
    print("    --pred_dir predictions/ \\")
    print("    --gt_dir ground_truth/ \\")
    print("    --results_json metrics_only.json \\")
    print("    --no_summary")
    
    print("\n" + "=" * 50)
    print("For actual usage, replace the example paths with your real data paths.")
    print("Make sure mask_comparison.py is in your current directory or in your PATH.")
