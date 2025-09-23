"""seg_rl - heatmap: Heatmap-based point localization training and inference.

Modules
-------
- datasets: Dataset and transforms for JSONL point supervision (supports new and legacy formats)
- model: Backbone and heatmap head producing per-pixel logits
- losses: CE over pixels and KL/MSE to Gaussian targets  
- utils: Coordinate utilities, metrics, visualization helpers

Data Formats
------------
New format (from gen_point_jsonl_from_masks.py):
    {"image": "/path/img.jpg", "points": [[x1,y1], [x2,y2]], "labels": [1, 0]}

Legacy format (backward compatible):
    {"image": "/path/img.jpg", "x": x1, "y": y1}
"""

__all__ = [
    "datasets",
    "model",
    "losses",
    "utils",
]


