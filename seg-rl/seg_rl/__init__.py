"""seg_rl: Heatmap-based point localization training and inference.

Modules
-------
- datasets: Dataset and transforms for {image, x, y} supervision
- model: Backbone and heatmap head producing per-pixel logits
- losses: CE over pixels and KL/MSE to Gaussian targets
- utils: Coordinate utilities, metrics, visualization helpers
"""

__all__ = [
    "datasets",
    "model",
    "losses",
    "utils",
]


