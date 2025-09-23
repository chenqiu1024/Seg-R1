#!/usr/bin/env python3
"""
Mask Comparison and Evaluation Script

This script compares predicted masks with ground truth masks and provides comprehensive evaluation metrics
and visualization capabilities.

Author: Generated for Seg-R1 project
Date: 2025-09-23
"""

import os
import argparse
import sys
from pathlib import Path
import random
from typing import Dict, List, Tuple, Optional
import json

# Check for required dependencies
try:
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from tqdm import tqdm
except ImportError as e:
    print(f"Error: Missing required dependency: {e}")
    print("\nPlease install required packages:")
    print("pip install opencv-python matplotlib numpy tqdm")
    print("\nOr install from requirements file:")
    print("pip install -r requirements_mask_comparison.txt")
    sys.exit(1)


class MaskEvaluator:
    """Class for evaluating mask predictions against ground truth"""
    
    def __init__(self):
        self.metrics = []
        
    def calculate_dice(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """Calculate DICE coefficient"""
        pred_mask = (pred_mask > 0).astype(np.uint8)
        gt_mask = (gt_mask > 0).astype(np.uint8)
        
        intersection = np.sum(pred_mask * gt_mask)
        union = np.sum(pred_mask) + np.sum(gt_mask)
        
        if union == 0:
            return 1.0 if np.sum(pred_mask) == 0 else 0.0
        
        dice = 2.0 * intersection / union
        return dice
    
    def calculate_iou(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """Calculate Intersection over Union (IoU)"""
        pred_mask = (pred_mask > 0).astype(np.uint8)
        gt_mask = (gt_mask > 0).astype(np.uint8)
        
        intersection = np.sum(pred_mask * gt_mask)
        union = np.sum(pred_mask) + np.sum(gt_mask) - intersection
        
        if union == 0:
            return 1.0 if np.sum(pred_mask) == 0 else 0.0
        
        iou = intersection / union
        return iou
    
    def calculate_precision(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """Calculate Precision"""
        pred_mask = (pred_mask > 0).astype(np.uint8)
        gt_mask = (gt_mask > 0).astype(np.uint8)
        
        true_positive = np.sum(pred_mask * gt_mask)
        predicted_positive = np.sum(pred_mask)
        
        if predicted_positive == 0:
            return 1.0 if np.sum(gt_mask) == 0 else 0.0
        
        precision = true_positive / predicted_positive
        return precision
    
    def calculate_recall(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """Calculate Recall"""
        pred_mask = (pred_mask > 0).astype(np.uint8)
        gt_mask = (gt_mask > 0).astype(np.uint8)
        
        true_positive = np.sum(pred_mask * gt_mask)
        actual_positive = np.sum(gt_mask)
        
        if actual_positive == 0:
            return 1.0 if np.sum(pred_mask) == 0 else 0.0
        
        recall = true_positive / actual_positive
        return recall
    
    def calculate_s_measure(self, pred_mask: np.ndarray, gt_mask: np.ndarray, alpha: float = 0.5) -> float:
        """Calculate S-measure (Structure measure)"""
        pred_mask = (pred_mask > 0).astype(np.float32)
        gt_mask = (gt_mask > 0).astype(np.float32)
        
        # Object-aware structural similarity
        def ssim_object(pred, gt):
            fg_pred = pred * gt
            fg_gt = gt
            
            if np.sum(fg_gt) == 0:
                return 1.0 if np.sum(fg_pred) == 0 else 0.0
            
            mean_pred = np.mean(fg_pred)
            mean_gt = np.mean(fg_gt)
            
            var_pred = np.var(fg_pred)
            var_gt = np.var(fg_gt)
            cov = np.mean((fg_pred - mean_pred) * (fg_gt - mean_gt))
            
            c1, c2 = 0.01, 0.03
            ssim = ((2 * mean_pred * mean_gt + c1) * (2 * cov + c2)) / \
                   ((mean_pred**2 + mean_gt**2 + c1) * (var_pred + var_gt + c2))
            return max(0, ssim)
        
        # Region-aware structural similarity  
        def ssim_region(pred, gt):
            h, w = pred.shape
            if h < 2 or w < 2:
                return np.mean(pred == gt)
            
            # Divide into 4 regions
            h_mid, w_mid = h // 2, w // 2
            regions = [
                (pred[:h_mid, :w_mid], gt[:h_mid, :w_mid]),
                (pred[:h_mid, w_mid:], gt[:h_mid, w_mid:]),
                (pred[h_mid:, :w_mid], gt[h_mid:, :w_mid]),
                (pred[h_mid:, w_mid:], gt[h_mid:, w_mid:])
            ]
            
            ssim_scores = []
            for pred_region, gt_region in regions:
                if pred_region.size == 0:
                    continue
                mean_pred = np.mean(pred_region)
                mean_gt = np.mean(gt_region)
                
                if mean_gt == 0 and mean_pred == 0:
                    ssim_scores.append(1.0)
                elif mean_gt == 0 or mean_pred == 0:
                    ssim_scores.append(0.0)
                else:
                    var_pred = np.var(pred_region)
                    var_gt = np.var(gt_region)
                    cov = np.cov(pred_region.flatten(), gt_region.flatten())[0, 1]
                    
                    c1, c2 = 0.01, 0.03
                    ssim = ((2 * mean_pred * mean_gt + c1) * (2 * cov + c2)) / \
                           ((mean_pred**2 + mean_gt**2 + c1) * (var_pred + var_gt + c2))
                    ssim_scores.append(max(0, ssim))
            
            return np.mean(ssim_scores) if ssim_scores else 0.0
        
        s_object = ssim_object(pred_mask, gt_mask)
        s_region = ssim_region(pred_mask, gt_mask)
        
        s_measure = alpha * s_object + (1 - alpha) * s_region
        return s_measure
    
    def evaluate_pair(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
        """Evaluate a single mask pair"""
        metrics = {
            'dice': self.calculate_dice(pred_mask, gt_mask),
            'iou': self.calculate_iou(pred_mask, gt_mask),
            'precision': self.calculate_precision(pred_mask, gt_mask),
            'recall': self.calculate_recall(pred_mask, gt_mask),
            's_measure': self.calculate_s_measure(pred_mask, gt_mask)
        }
        
        # Calculate F1-score
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1_score'] = 0.0
            
        return metrics


class MaskVisualizer:
    """Class for visualizing mask comparisons"""
    
    def __init__(self, pred_color=(1.0, 0.0, 0.0), gt_color=(0.0, 1.0, 0.0), alpha=0.5):
        self.pred_color = pred_color  # Red for predictions
        self.gt_color = gt_color      # Green for ground truth
        self.alpha = alpha
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load image and convert to grayscale if needed"""
        if not os.path.exists(image_path):
            return None
        
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to grayscale if color image
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        return img
    
    def create_overlay_visualization(self, pred_mask: np.ndarray, gt_mask: np.ndarray, 
                                   background_img: Optional[np.ndarray] = None,
                                   metrics: Optional[Dict[str, float]] = None,
                                   title: str = "") -> np.ndarray:
        """Create overlay visualization of masks"""
        
        # Normalize masks
        pred_mask = (pred_mask > 0).astype(np.uint8)
        gt_mask = (gt_mask > 0).astype(np.uint8)
        
        # Get image dimensions
        h, w = pred_mask.shape
        
        # Create base image
        if background_img is not None:
            if background_img.shape[:2] != (h, w):
                background_img = cv2.resize(background_img, (w, h))
            # Normalize background to 0-1 range
            base_img = background_img.astype(np.float32) / 255.0
            if len(base_img.shape) == 2:
                base_img = np.stack([base_img] * 3, axis=-1)
            # Darken background for better mask visibility
            base_img = base_img * 0.6
        else:
            base_img = np.zeros((h, w, 3), dtype=np.float32)
        
        # Create colored masks
        pred_colored = np.zeros((h, w, 3), dtype=np.float32)
        gt_colored = np.zeros((h, w, 3), dtype=np.float32)
        
        # Apply colors
        pred_colored[pred_mask > 0] = self.pred_color
        gt_colored[gt_mask > 0] = self.gt_color
        
        # Combine masks with background
        overlay = base_img.copy()
        overlay = overlay + self.alpha * pred_colored
        overlay = overlay + self.alpha * gt_colored
        
        # Clip values
        overlay = np.clip(overlay, 0, 1)
        
        return overlay
    
    def visualize_single_pair(self, pred_mask: np.ndarray, gt_mask: np.ndarray,
                            background_img: Optional[np.ndarray] = None,
                            metrics: Optional[Dict[str, float]] = None,
                            title: str = "",
                            save_path: Optional[str] = None) -> None:
        """Visualize a single mask pair"""
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original masks
        axes[0].imshow(pred_mask, cmap='Reds', alpha=0.7)
        if background_img is not None:
            axes[0].imshow(background_img, cmap='gray', alpha=0.3)
        axes[0].set_title('Predicted Mask')
        axes[0].axis('off')
        
        axes[1].imshow(gt_mask, cmap='Greens', alpha=0.7)
        if background_img is not None:
            axes[1].imshow(background_img, cmap='gray', alpha=0.3)
        axes[1].set_title('Ground Truth Mask')
        axes[1].axis('off')
        
        # Overlay
        overlay = self.create_overlay_visualization(pred_mask, gt_mask, background_img, metrics)
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay Comparison')
        axes[2].axis('off')
        
        # Add legend
        red_patch = patches.Patch(color='red', alpha=0.5, label='Predicted')
        green_patch = patches.Patch(color='green', alpha=0.5, label='Ground Truth')
        axes[2].legend(handles=[red_patch, green_patch], loc='upper right')
        
        # Add metrics text
        if metrics is not None:
            metrics_text = []
            for key, value in metrics.items():
                metrics_text.append(f"{key.upper()}: {value:.3f}")
            
            fig.text(0.02, 0.02, '\n'.join(metrics_text), fontsize=10, 
                    verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def create_summary_visualization(self, mask_pairs: List[Tuple], metrics_list: List[Dict],
                                   num_samples: int = 6, save_path: Optional[str] = None) -> None:
        """Create summary visualization with multiple mask pairs"""
        
        # Randomly sample pairs if there are too many
        if len(mask_pairs) > num_samples:
            indices = random.sample(range(len(mask_pairs)), num_samples)
            mask_pairs = [mask_pairs[i] for i in indices]
            metrics_list = [metrics_list[i] for i in indices]
        
        cols = 3
        rows = (len(mask_pairs) + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        if cols == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, ((pred_mask, gt_mask, bg_img, name), metrics) in enumerate(zip(mask_pairs, metrics_list)):
            row, col = idx // cols, idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            overlay = self.create_overlay_visualization(pred_mask, gt_mask, bg_img, metrics)
            ax.imshow(overlay)
            
            # Add metrics text
            metrics_text = f"DICE: {metrics['dice']:.3f}\nIoU: {metrics['iou']:.3f}\nS-measure: {metrics['s_measure']:.3f}"
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=8,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_title(f"{name}", fontsize=10)
            ax.axis('off')
        
        # Hide empty subplots
        for idx in range(len(mask_pairs), rows * cols):
            row, col = idx // cols, idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.axis('off')
        
        # Add overall legend
        red_patch = patches.Patch(color='red', alpha=0.5, label='Predicted')
        green_patch = patches.Patch(color='green', alpha=0.5, label='Ground Truth')
        fig.legend(handles=[red_patch, green_patch], loc='lower center', ncol=2, fontsize=12)
        
        plt.suptitle('Mask Comparison Summary', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def find_matching_files(pred_dir: str, gt_dir: str, img_dir: Optional[str] = None) -> List[Tuple[str, str, Optional[str]]]:
    """Find matching files between prediction and ground truth directories"""
    pred_files = set(os.listdir(pred_dir))
    gt_files = set(os.listdir(gt_dir))
    
    # Find common files
    common_files = pred_files.intersection(gt_files)
    
    matching_files = []
    for filename in sorted(common_files):
        pred_path = os.path.join(pred_dir, filename)
        gt_path = os.path.join(gt_dir, filename)
        
        # Look for corresponding image file
        img_path = None
        if img_dir:
            # Try different extensions
            base_name = os.path.splitext(filename)[0]
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                potential_img = os.path.join(img_dir, base_name + ext)
                if os.path.exists(potential_img):
                    img_path = potential_img
                    break
        
        matching_files.append((pred_path, gt_path, img_path))
    
    return matching_files


def load_mask(mask_path: str) -> np.ndarray:
    """Load mask image"""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot load mask from {mask_path}")
    return mask


def compute_statistics(metrics_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Compute statistics for all metrics"""
    if not metrics_list:
        return {}
    
    stats = {}
    metric_names = metrics_list[0].keys()
    
    for metric_name in metric_names:
        values = [m[metric_name] for m in metrics_list]
        stats[metric_name] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values)
        }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Compare predicted masks with ground truth masks')
    parser.add_argument('--pred_dir', type=str, required=True, help='Directory containing predicted masks')
    parser.add_argument('--gt_dir', type=str, required=True, help='Directory containing ground truth masks')
    parser.add_argument('--img_dir', type=str, help='Directory containing original images (optional)')
    parser.add_argument('--output_dir', type=str, help='Output directory for individual comparison images')
    parser.add_argument('--summary_output', type=str, help='Path for summary visualization')
    parser.add_argument('--summary_samples', type=int, default=6, help='Number of samples in summary visualization')
    parser.add_argument('--results_json', type=str, help='Path to save detailed results as JSON')
    parser.add_argument('--visualize_all', action='store_true', help='Generate visualization for all mask pairs')
    parser.add_argument('--no_summary', action='store_true', help='Skip summary visualization')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducible sampling')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # Validate input directories
    if not os.path.exists(args.pred_dir):
        raise ValueError(f"Prediction directory does not exist: {args.pred_dir}")
    if not os.path.exists(args.gt_dir):
        raise ValueError(f"Ground truth directory does not exist: {args.gt_dir}")
    if args.img_dir and not os.path.exists(args.img_dir):
        print(f"Warning: Image directory does not exist: {args.img_dir}")
        args.img_dir = None
    
    # Create output directory if needed
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Find matching files
    print("Finding matching files...")
    matching_files = find_matching_files(args.pred_dir, args.gt_dir, args.img_dir)
    print(f"Found {len(matching_files)} matching file pairs")
    
    if len(matching_files) == 0:
        print("No matching files found. Please check your directory paths.")
        return
    
    # Initialize evaluator and visualizer
    evaluator = MaskEvaluator()
    visualizer = MaskVisualizer()
    
    # Process all mask pairs
    all_metrics = []
    mask_pairs_for_summary = []
    
    print("Processing mask pairs...")
    for i, (pred_path, gt_path, img_path) in enumerate(tqdm(matching_files)):
        try:
            # Load masks
            pred_mask = load_mask(pred_path)
            gt_mask = load_mask(gt_path)
            
            # Load background image if available
            bg_img = None
            if img_path:
                bg_img = visualizer.load_image(img_path)
            
            # Evaluate metrics
            metrics = evaluator.evaluate_pair(pred_mask, gt_mask)
            all_metrics.append(metrics)
            
            # Store for summary visualization
            filename = os.path.basename(pred_path)
            mask_pairs_for_summary.append((pred_mask, gt_mask, bg_img, filename))
            
            # Generate individual visualization if requested
            if args.visualize_all and args.output_dir:
                output_path = os.path.join(args.output_dir, f"comparison_{os.path.splitext(filename)[0]}.png")
                visualizer.visualize_single_pair(
                    pred_mask, gt_mask, bg_img, metrics,
                    title=f"Comparison: {filename}",
                    save_path=output_path
                )
            
        except Exception as e:
            print(f"Error processing {pred_path}: {str(e)}")
            continue
    
    if len(all_metrics) == 0:
        print("No valid mask pairs were processed.")
        return
    
    # Compute statistics
    print("\nComputing statistics...")
    stats = compute_statistics(all_metrics)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Total processed pairs: {len(all_metrics)}")
    print()
    
    for metric_name, metric_stats in stats.items():
        print(f"{metric_name.upper()}:")
        print(f"  Mean: {metric_stats['mean']:.4f} ± {metric_stats['std']:.4f}")
        print(f"  Min:  {metric_stats['min']:.4f}")
        print(f"  Max:  {metric_stats['max']:.4f}")
        print(f"  Median: {metric_stats['median']:.4f}")
        print()
    
    # Save detailed results
    if args.results_json:
        detailed_results = {
            'statistics': stats,
            'individual_results': []
        }
        
        for i, (metrics, (pred_path, gt_path, _)) in enumerate(zip(all_metrics, matching_files)):
            detailed_results['individual_results'].append({
                'pred_file': os.path.basename(pred_path),
                'gt_file': os.path.basename(gt_path),
                'metrics': metrics
            })
        
        with open(args.results_json, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        print(f"Detailed results saved to: {args.results_json}")
    
    # Generate summary visualization
    if not args.no_summary:
        print("Generating summary visualization...")
        summary_path = args.summary_output or "mask_comparison_summary.png"
        visualizer.create_summary_visualization(
            mask_pairs_for_summary, all_metrics,
            num_samples=min(args.summary_samples, len(mask_pairs_for_summary)),
            save_path=summary_path
        )
        print(f"Summary visualization saved to: {summary_path}")
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
