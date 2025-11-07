"""
PEFT (Parameter Efficient Fine-Tuning) module for SAM2 segmentation.

This package implements Late LoRA fine-tuning for SAM2's image encoder,
combined with a new point prediction network that takes SAM features as input.
"""

from .lora_sam2 import LoRASAM2Wrapper, LoRALinear
from .point_predictor_peft import PointPredictorFromSAMFeatures
from .datasets_peft import PEFTPointDataset

__all__ = [
    'LoRASAM2Wrapper',
    'LoRALinear',
    'PointPredictorFromSAMFeatures',
    'PEFTPointDataset',
]

