import os
import re
import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import cv2
import numpy as np
import torch
import datasets
from PIL import Image as PILImage
from datasets import load_dataset
from torchvision.transforms import Resize
import torchvision.transforms.functional as TF
from transformers import Qwen2VLForConditionalGeneration, TrainingArguments
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainerModified
from utils.metrics import Smeasure
import warnings
warnings.filterwarnings("ignore")

RESIZE_SIZE = (768, 768)
_TYPE = np.float64  


def resize_image_and_mask(example: dict) -> dict:
    """Resize input image and corresponding mask to specified dimensions.
    
    Args:
        example: Dictionary containing 'image' and 'gt_path' keys
        
    Returns:
        Dictionary with resized 'image' (PIL Image) and 'gt_mask' (numpy array)
    """
    image = example["image"].convert("RGB")
    image_resized = TF.resize(image, RESIZE_SIZE)

    # Load and process mask
    mask = PILImage.open(example["gt_path"]).convert("L")
    mask_resized = TF.resize(mask, RESIZE_SIZE)
    gt_mask_np = (np.array(mask_resized) > 127).astype(np.uint8)  # Binarize mask

    return {
        "image": image_resized,
        "gt_mask": gt_mask_np
    }


class SAMWrapper:
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """Initialize SAM2 model and predictor.
        
        Args:
            model_path: Path to SAM2 model checkpoint
            device: Device to run model on (e.g. "cuda", "cuda:0", "cpu"). 
                   If None, will auto-detect available device.
        """
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.model = build_sam2(model_cfg, model_path)
        
        # Device configuration
        self.device = torch.device(
            device if device else 
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)
        
        self.predictor = SAM2ImagePredictor(self.model)
        self.last_mask = None
        
    def predict(self, 
               image: PILImage.Image, 
               points: List[Tuple[int, int]], 
               labels: List[int]) -> Tuple[np.ndarray, float]:
        """Run segmentation prediction with given prompts.
        
        Args:
            image: Input PIL Image
            points: List of (x,y) coordinate points
            labels: List of point labels (1=foreground, 0=background)
            
        Returns:
            Tuple of (predicted_mask, confidence_score)
        """
        # Convert inputs to numpy arrays
        input_points = np.array(points)
        input_labels = np.array(labels)

        # Convert and preprocess image
        image_np = np.array(image)
        rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        
        # Run prediction
        self.predictor.set_image(rgb_image)
        mask_pred, score, _ = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False,
        )
        
        self.last_mask = mask_pred[0]
        return mask_pred[0], score[0]


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """Extended training arguments for GRPO segmentation."""
    
    reward_funcs: List[str] = field(
        default_factory=lambda: ["segmentation"],
        metadata={"help": "List of reward functions to use"}
    )
    sam_checkpoint: str = field(
        default="third_party/sam2/checkpoints/sam2.1_hiera_large.pt",
        metadata={"help": "Path to SAM2 model checkpoint"}
    )
    dataset_image: str = field(
        default=None,
        metadata={"help": "Directory containing training images"}
    )
    dataset_gt: str = field(
        default=None,
        metadata={"help": "Directory containing ground truth masks"}
    )
    dataset_train_split: str = field(
        default="train",
        metadata={"help": "Dataset split to use for training"}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "If set, limit the maximum number of input samples loaded from dataset_image."}
    )
    sam_device: str = field(
        default=None,
        metadata={"help": "Device to use (e.g. 'cuda:0', 'cpu'). Auto-detects if None."}
    )


def parse_custom_format(content: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Parse point coordinates and labels from formatted string.
    
    Args:
        content: String containing formatted points and labels
        
    Returns:
        Tuple of (points_array, labels_array) or (None, None) if parsing fails
    """
    point_pattern = r"<points>\s*(\[\s*(?:\[\s*\d+\s*,\s*\d+\s*\]\s*,?\s*)+\])\s*</points>"
    label_pattern = r"<labels>\s*(\[\s*(?:\d+\s*,?\s*)+\])\s*</labels>"
    
    point_match = re.search(point_pattern, content)
    label_match = re.search(label_pattern, content)
    
    try:
        if point_match and label_match:
            points = np.array(eval(point_match.group(1)))
            labels = np.array(eval(label_match.group(1)))
            
            # Validate dimensions
            if (len(points.shape) == 2 and 
                points.shape[1] == 2 and
                len(labels) == points.shape[0]):
                return points, labels
    except Exception:
        pass
        
    return None, None


def format_reward(completions: List[dict], **kwargs) -> List[float]:
    """Calculate reward based on format compliance.
    
    Args:
        completions: List of model completion dictionaries
        
    Returns:
        List of reward scores (1.0 for valid format, 0.0 otherwise)
    """
    return [
        1.0 if parse_custom_format(c[0]["content"])[0] is not None else 0.0
        for c in completions
    ]


def _prepare_data(pred: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize prediction and ground truth data formats.
    
    Args:
        pred: Prediction array
        gt: Ground truth array
        
    Returns:
        Tuple of normalized (prediction, ground_truth)
    """
    pred = pred.astype(_TYPE)
    gt = gt.astype(_TYPE)
    return pred, (gt > 128).astype(_TYPE)


def segmentation_reward(
    completions: List[str],
    gt_mask: List[np.ndarray],
    image: List[PILImage.Image],
    **kwargs
) -> List[float]:
    """Calculate segmentation quality reward.
    
    Args:
        completions: List of model completion strings
        gt_mask: List of ground truth masks
        image: List of input images
        
    Returns:
        List of reward scores combining IOU and S-measure metrics
    """
    rewards = []
    sm_calculator = Smeasure(alpha=kwargs.get('s_measure_alpha', 0.5))
    sam_config = kwargs.get("sam_config")[0]
    sam = SAMWrapper(
            model_path=sam_config["model_path"],
            device=sam_config["device"]
        )
    
    for completion, gt_mask, img in zip(completions, gt_mask, image):
        content = completion[0]["content"]
        points, labels = parse_custom_format(content)
        # print(f"Parsed points: {points}, labels: {labels}")

        iou_reward = 0.0
        sm_reward = 0.0
        # point_penalty = 0.0
        
        if points is not None and len(points) > 0:

            if not isinstance(img, PILImage.Image):
                img = PILImage.fromarray(img)
            
            mask_pred, score = sam.predict(img, points.tolist(), labels.tolist())
            # pred_mask = PILImage.fromarray((mask_pred * 255).astype(np.uint8))
            # pred_mask_path = "pred_mask.png"
            # pred_mask.save(pred_mask_path)            
            intersection = np.logical_and(mask_pred, gt_mask).sum()
            union = np.logical_or(mask_pred, gt_mask).sum()
            iou_reward = intersection / union if union > 0 else 0.0

            mask_pred = np.array(mask_pred).astype(_TYPE)
            gt_np = np.array(gt_mask).astype(_TYPE)
            mask_pred, gt_np = _prepare_data(mask_pred, gt_np)
            
            sm = sm_calculator.cal_sm(mask_pred, gt_np)
            sm_reward = max(0.0, min(1.0, sm))  
        total_reward = 0.7*iou_reward  + 0.3*sm_reward 
        rewards.append(total_reward)
        
        if os.getenv("DEBUG") == "1":
            log_str = f"""
            Content: {content}
            Points: {points}
            Labels: {labels}
            IOU: {iou_reward:.2f}
            Final Reward: {total_reward:.2f}
            {'-'*40}
            """
            with open("training.log", "a") as f:
                f.write(log_str)
    
    return rewards


def create_conversation(example: dict) -> dict:
    """Create training prompt with instructions for segmentation task."""
    return {
        "prompt": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": 
                        """
You are given an image. Your task is to generate a sparse set of point prompts to help segment the most salient object in the image.

The points should be informative: include both foreground points (on the object) and background points (off the object) to help define the object boundary.

Only generate a moderate number of points, enough to clearly outline the object without overloading the system.

Output the result using the exact format:
<points>[[x1,y1],[x2,y2],...]</points> <labels>[1,0,...]</labels>
Where 1 indicates a foreground (object) point, and 0 indicates a background point.

"""
                        # "Generate sparse prompt points for to generate a detailed mask for the salient object in this image. "
                        # "Output coordinates in <points>[[x1,y1],[x2,y2],...]</points> "
                        # "and labels in <labels>[1,0,...]</labels>, (0 for background, 1 for object). "
                        # "e.g. <points>[[100,200],[150,250],[151,252],[232,132],[2,2],[10,0]]</points> <labels>[1,0,1,1,0,0]</labels>"
                    }
                ],
            }
        ],
        # "gt_mask": example["mask"],
        # "image": example["image"],
    }

class SegR1Trainer(Qwen2VLGRPOVLLMTrainerModified):
    
    def __init__(self, *args, sam_config=None, **kwargs):
        super().__init__(*args, **kwargs)



# Reward function registry
REWARD_FUNCS = {
    "segment": segmentation_reward,
    "format": format_reward,
}


def main(script_args: GRPOScriptArguments, 
        training_args: TrainingArguments, 
        model_args: ModelConfig) -> None:
    """Main training pipeline.
    
    Args:
        script_args: Script configuration arguments
        training_args: Training hyperparameters
        model_args: Model configuration
    """
    # Initialize reward functions
    reward_funcs = [REWARD_FUNCS[func] for func in ["segment", "format"]]
    
    # Load and prepare dataset
    dataset = load_dataset(
        "imagefolder",
        data_dir=script_args.dataset_image,
        split=script_args.dataset_train_split,
    )
    # Optionally cap the number of samples used for training
    if script_args.max_train_samples is not None:
        max_n = int(script_args.max_train_samples)
        if max_n > 0:
            dataset = dataset.select(range(min(max_n, len(dataset))))
    
    # Add ground truth paths and resize
    dataset = dataset.map(
        lambda x: {"gt_path": os.path.join(
            script_args.dataset_gt,
            os.path.basename(x["image"].filename).replace(".jpg", ".png")
        )
        }
    ).map(resize_image_and_mask)
    dataset = dataset.map(
        lambda x: {
            "sam_config": {
                "device": script_args.sam_device,
                "model_path": script_args.sam_checkpoint
            }
        }
    )
    # Create training conversations
    if "image" not in dataset.features:
        raise ValueError("Dataset missing required 'image' field")
    dataset = dataset.map(create_conversation)

    # Choose trainer class based on whether vLLM is enabled
    TrainerClass = Qwen2VLGRPOVLLMTrainerModified if getattr(training_args, "use_vllm", False) else Qwen2VLGRPOTrainer

    # For single-GPU (non-vLLM), ensure lower-precision weights to reduce memory
    if not getattr(training_args, "use_vllm", False):
        if not hasattr(training_args, "model_init_kwargs") or training_args.model_init_kwargs is None:
            training_args.model_init_kwargs = {}
        training_args.model_init_kwargs.setdefault("torch_dtype", "bfloat16")

    # Initialize and run trainer
    trainer = TrainerClass(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
    )
    
    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)