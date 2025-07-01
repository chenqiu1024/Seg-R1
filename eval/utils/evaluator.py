"""
Evaluation script for segmentation models.

This script calculates various metrics (S-measure, F-measure, MAE, etc.) 
to evaluate segmentation model performance against ground truth data.
"""

import os
import argparse
from glob import glob
from typing import List, Dict, Tuple

import prettytable as pt
from utils.metrics import evaluator


class SegmentationEvaluator:
    """Handles evaluation of segmentation models against ground truth."""
    
    METRIC_NAMES = ["S", "MAE", "E", "F", "WF", "MBA", "BIoU", "MSE"]
    
    def __init__(self, gt_root: str, pred_root: str, model_lst: List[str] = None, 
                 data_lst: str = "COD", save_dir: str = "eval/logs", 
                 check_integrity: bool = False):
        """Initialize evaluator with configuration.
        
        Args:
            gt_root: Path to ground truth directory
            pred_root: Path to predictions directory
            model_lst: List of model names to evaluate
            data_lst: Dataset names separated by '+'
            save_dir: Directory to save results
            check_integrity: Whether to verify file correspondence
        """
        self.gt_root = gt_root
        self.pred_root = pred_root
        self.model_lst = model_lst or ["Seg-R1"]
        self.data_lst = data_lst
        self.save_dir = save_dir
        self.check_integrity = check_integrity
        self.metrics = '+'.join(self.METRIC_NAMES)
        
        os.makedirs(self.save_dir, exist_ok=True)
        
    def _check_data_integrity(self) -> None:
        """Verify ground truth and prediction files match."""
        for data_name in self.data_lst.split('+'):
            for model_name in self.model_lst:
                gt_path = self.gt_root
                pred_path = self.pred_root
                
                gt_files = sorted(os.listdir(gt_path))
                pred_files = sorted(os.listdir(pred_path))
                
                if gt_files != pred_files:
                    print(f"Mismatch in {data_name} for {model_name}:")
                    print(f"GT files: {len(gt_files)}, Pred files: {len(pred_files)}")

    def _format_score(self, score: float) -> str:
        """Format score for consistent display."""
        if score <= 1:
            return f".{score:.3f}".split('.')[-1]
        return f"{score:<4}"

    def _get_metrics(self) -> str:
        """Determine which metrics to calculate based on dataset."""
        if any('DIS-' in data for data in self.data_lst.split('+')):
            return '+'.join(self.METRIC_NAMES[:100])  # All metrics
        return '+'.join(self.METRIC_NAMES[:-1])  # Exclude last metric

    def _evaluate_dataset(self, data_name: str) -> pt.PrettyTable:
        """Evaluate models on a specific dataset."""
        print('#' * 20, data_name, '#' * 20)
        
        # Initialize results table
        table = pt.PrettyTable()
        table.vertical_char = '&'
        table.field_names = [
            "Dataset", "Method", "Smeasure", "wFmeasure", "meanFm", 
            "meanEm", "maxEm", "MAE", "maxFm", "adpEm", "adpFm", 
            "HCE", "mBA", "maxBIoU", "meanBIoU"
        ]
        
        # Get ground truth paths
        gt_paths = sorted(glob(os.path.join(self.gt_root, '*.png')))
        
        for model_name in self.model_lst:
            print(f"\tEvaluating model: {model_name}...")
            
            # Get prediction paths
            pred_paths = [
                p.replace(self.gt_root, self.pred_root)
                for p in gt_paths
            ]
            
            # Calculate metrics
            metrics = evaluator(
                gt_paths=gt_paths,
                pred_paths=pred_paths,
                metrics=self.metrics.split('+'),
                verbose=True,
            )
            
            # Format results
            scores = self._format_metrics(*metrics)
            table.add_row([data_name, model_name] + scores)
            
            # Save intermediate results
            result_file = os.path.join(self.save_dir, f'{data_name}_eval.txt')
            with open(result_file, 'w+') as f:
                f.write(str(table) + '\n')
        
        return table

    def _format_metrics(self, em, sm, fm, mae, mse, wfm, hce, mba, biou) -> List[str]:
        """Format metric results for display."""
        return [
            self._format_score(metric) 
            for metric in [
                sm.round(3), wfm.round(3), fm['curve'].mean().round(3),
                em['curve'].mean().round(3), em['curve'].max().round(3),
                mae.round(3), fm['curve'].max().round(3),
                em['adp'].round(3), fm['adp'].round(3), int(hce.round()),
                mba.round(3), biou['curve'].max().round(3),
                biou['curve'].mean().round(3)
            ]
        ]

    def run(self) -> None:
        """Run full evaluation pipeline."""
        if self.check_integrity:
            self._check_data_integrity()
        else:
            print('>>> Skipping file integrity check')
        
        for data_name in self.data_lst.split('+'):
            result_table = self._evaluate_dataset(data_name)
            print(result_table)
    def evaluate(self) -> Dict[str, pt.PrettyTable]:
        """Run evaluation and return results.
        
        Returns:
            Dictionary mapping dataset names to PrettyTable results
        """
        results = {}
        if self.check_integrity:
            self._check_data_integrity()
        
        for data_name in self.data_lst.split('+'):
            results[data_name] = self._evaluate_dataset(data_name)
        
        return results