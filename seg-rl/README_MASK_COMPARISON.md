# Mask Comparison and Evaluation Tool

This tool provides comprehensive evaluation and visualization capabilities for comparing predicted segmentation masks with ground truth masks.

## Features

- **Multiple Evaluation Metrics**: DICE, IoU, Precision, Recall, F1-score, S-measure
- **Flexible Visualization**: Individual comparisons and summary visualizations
- **Background Image Support**: Overlay masks on original images for better context
- **Statistical Analysis**: Mean, standard deviation, min, max, and median values
- **Batch Processing**: Process entire directories of mask pairs
- **Export Capabilities**: Save results as JSON and visualization images

## Installation

Required dependencies:
```bash
pip install opencv-python matplotlib numpy tqdm
```

## Usage

### Basic Usage

Compare prediction masks with ground truth masks:
```bash
python mask_comparison.py --pred_dir /path/to/predictions --gt_dir /path/to/ground_truth
```

### Advanced Usage

Full feature example with all options:
```bash
python mask_comparison.py \
    --pred_dir /path/to/predictions \
    --gt_dir /path/to/ground_truth \
    --img_dir /path/to/original_images \
    --output_dir /path/to/individual_comparisons \
    --summary_output summary_comparison.png \
    --results_json detailed_results.json \
    --summary_samples 8 \
    --visualize_all \
    --random_seed 42
```

## Command Line Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--pred_dir` | ✓ | Directory containing predicted masks |
| `--gt_dir` | ✓ | Directory containing ground truth masks |
| `--img_dir` | | Directory containing original images (optional) |
| `--output_dir` | | Output directory for individual comparison images |
| `--summary_output` | | Path for summary visualization (default: mask_comparison_summary.png) |
| `--summary_samples` | | Number of samples in summary visualization (default: 6) |
| `--results_json` | | Path to save detailed results as JSON |
| `--visualize_all` | | Generate visualization for all mask pairs |
| `--no_summary` | | Skip summary visualization |
| `--random_seed` | | Random seed for reproducible sampling (default: 42) |

## File Organization

The tool expects the following file organization:

```
predictions/
├── image1.png
├── image2.png
└── image3.png

ground_truth/
├── image1.png
├── image2.png
└── image3.png

original_images/  (optional)
├── image1.jpg
├── image2.jpg
└── image3.jpg
```

**Important**: Prediction and ground truth masks must have **identical filenames**. Original images should have the same base filename but can have different extensions (.jpg, .jpeg, .png, .bmp, .tiff).

## Evaluation Metrics

### 1. DICE Coefficient
Measures overlap between predicted and ground truth masks:
```
DICE = 2 * |Pred ∩ GT| / (|Pred| + |GT|)
```

### 2. Intersection over Union (IoU)
Measures ratio of intersection to union:
```
IoU = |Pred ∩ GT| / |Pred ∪ GT|
```

### 3. Precision
Measures accuracy of positive predictions:
```
Precision = True Positives / (True Positives + False Positives)
```

### 4. Recall
Measures sensitivity/coverage:
```
Recall = True Positives / (True Positives + False Negatives)
```

### 5. F1-Score
Harmonic mean of precision and recall:
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

### 6. S-measure
Structure-aware similarity measure that considers both object-aware and region-aware structural similarity.

## Visualization Features

### Individual Comparisons
- Side-by-side view of predicted mask, ground truth, and overlay
- Metrics displayed as text annotations
- Optional background image integration
- Color-coded masks (red for predictions, green for ground truth)

### Summary Visualization
- Grid layout showing multiple mask pairs
- Random sampling for large datasets
- Compact metrics display
- Consistent color scheme

### Color Scheme
- **Red (semi-transparent)**: Predicted masks
- **Green (semi-transparent)**: Ground truth masks
- **Yellow/Orange**: Overlapping regions
- **Grayscale background**: Original images (when provided)

## Output Files

### Statistics Output (Console)
```
EVALUATION RESULTS
==================
Total processed pairs: 150

DICE:
  Mean: 0.8234 ± 0.1456
  Min:  0.2341
  Max:  0.9876
  Median: 0.8567

IoU:
  Mean: 0.7123 ± 0.1678
  ...
```

### JSON Results File
```json
{
  "statistics": {
    "dice": {
      "mean": 0.8234,
      "std": 0.1456,
      "min": 0.2341,
      "max": 0.9876,
      "median": 0.8567
    },
    ...
  },
  "individual_results": [
    {
      "pred_file": "image1.png",
      "gt_file": "image1.png",
      "metrics": {
        "dice": 0.8456,
        "iou": 0.7321,
        ...
      }
    },
    ...
  ]
}
```

## Example Workflows

### 1. Quick Evaluation
Just get the statistics without visualization:
```bash
python mask_comparison.py \
    --pred_dir predictions/ \
    --gt_dir ground_truth/ \
    --no_summary
```

### 2. Detailed Analysis with Original Images
```bash
python mask_comparison.py \
    --pred_dir predictions/ \
    --gt_dir ground_truth/ \
    --img_dir original_images/ \
    --output_dir individual_results/ \
    --summary_output overall_summary.png \
    --results_json evaluation_results.json \
    --visualize_all
```

### 3. Custom Summary Visualization
```bash
python mask_comparison.py \
    --pred_dir predictions/ \
    --gt_dir ground_truth/ \
    --summary_output custom_summary.png \
    --summary_samples 12 \
    --random_seed 123
```

## Error Handling

The tool handles various error conditions gracefully:
- Missing files (skipped with warning)
- Corrupted images (skipped with error message)
- Mismatched image dimensions (automatically resized)
- Empty directories (informative error message)

## Performance Considerations

- Processing time scales linearly with number of mask pairs
- Memory usage depends on image sizes and visualization options
- For large datasets (>1000 images), consider:
  - Using `--no_summary` to skip summary visualization
  - Avoiding `--visualize_all` unless necessary
  - Processing in batches

## Troubleshooting

### Common Issues

1. **No matching files found**
   - Check that filenames match exactly between directories
   - Ensure correct directory paths
   - Verify file extensions are supported

2. **Memory errors with large images**
   - Reduce image resolution before processing
   - Process in smaller batches
   - Use `--no_summary` and `--visualize_all` false

3. **Poor S-measure calculation**
   - Ensure masks are properly binarized
   - Check for edge cases with empty masks
   - Verify image preprocessing steps

### Debug Tips

- Use small test datasets first
- Check individual mask files manually
- Enable verbose output for detailed error messages
- Verify input data format and range (0-255 for masks)

## License

This tool is part of the Seg-R1 project. Please refer to the main project license for usage terms.
