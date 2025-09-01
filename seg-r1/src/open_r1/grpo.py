import os
import re
import math
from math import exp
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import cv2
import numpy as np
import torch
import datasets
from PIL import Image as PILImage
from datasets import DatasetDict, load_dataset, concatenate_datasets
from torchvision.transforms import Resize
import torchvision.transforms.functional as TF
from transformers import Qwen2VLForConditionalGeneration, TrainingArguments
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainerModified
from utils.metrics import Smeasure


RESIZE_SIZE = (768, 768)
_TYPE = np.float64


def resize_image_and_mask(example):
    """Resize input image and corresponding mask to specified dimensions.
    
    Args:
        example: Dictionary containing 'image' and 'gt_path' keys
        
    Returns:
        Dictionary with resized 'image' (PIL Image) and 'gt_mask' (numpy array)
    """
    image = example["image"].convert("RGB")
    image_resized = TF.resize(image, RESIZE_SIZE)

    # Load and resize mask
    mask = PILImage.open(example["gt_path"]).convert("L")
    mask_resized = TF.resize(mask, RESIZE_SIZE)
    
    # Convert to numpy array for reward calculation
    gt_mask_np = np.array(mask_resized) > 127  # Binarize

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
        sam_model = build_sam2(model_cfg, model_path)
        
        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)
        
        # Move to specified device
        sam_model = sam_model.to(self.device)
        
        # Initialize predictor
        self.predictor = SAM2ImagePredictor(sam_model)
        self.last_mask = None
        
    def predict(self, 
               image: PILImage.Image, 
               points: List[Tuple[int, int]], 
               labels: List[int],
               bbox: Optional[List[List[int]]] = None) -> Tuple[np.ndarray, float]:
        """Run SAM2 prediction with given prompts.
        
        Args:
            image: Input PIL Image
            points: List of (x,y) point coordinates
            labels: List of point labels (1=foreground, 0=background)
            bbox: Optional bounding box [x1,y1,x2,y2]
            
        Returns:
            Tuple of (predicted_mask, confidence_score)
        """

        input_points = np.array(points) if points else None
        input_labels = np.array(labels) if labels else None
        input_bboxes = np.array(bbox) if bbox else None

        image_np = np.array(image)
        
        rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        
        self.predictor.set_image(rgb_image)
        
        mask_pred, score, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=input_bboxes,
            # previous_mask=self.last_mask,
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


def parse_custom_format(content: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Parse custom formatted string to extract points, labels and bbox.
    
    Supported format:
    <points>[[x1,y1],[x2,y2],...]</points>
    <labels>[1,0,...]</labels>
    <bbox>[x1,y1,x2,y2]</bbox>
    
    Args:
        content: String containing the formatted data
        
    Returns:
        Tuple of (points, labels, bbox) as numpy arrays
    """
    point_pattern = r"<points>\s*(\[\s*(?:\[\s*\d+\s*,\s*\d+\s*\]\s*,?\s*)+\])\s*</points>"
    label_pattern = r"<labels>\s*(\[\s*(?:\d+\s*,?\s*)+\])\s*</labels>"
    bbox_pattern  = r"<bbox>\s*(\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\])\s*</bbox>"

    point_match = re.search(point_pattern, content)
    label_match = re.search(label_pattern, content)
    bbox_matches = re.findall(bbox_pattern, content)

    try:
        points = np.array(eval(point_match.group(1))) if point_match else None
        labels = np.array(eval(label_match.group(1))) if label_match else None

        if points is not None and labels is not None:
            if not (len(points.shape) == 2 and points.shape[1] == 2 and len(labels) == points.shape[0]):
                points, labels = None, None

        bboxes = []
        for bbox_str in bbox_matches:
            bbox = np.array(eval(bbox_str))
            if len(bbox.shape) == 1 and bbox.shape[0] == 4:
                bboxes.append(bbox)
        
        bboxes = np.stack(bboxes, axis=0) if bboxes else None

        return points, labels, bboxes

    except Exception as e:
        print("Error parsing content:", e)
        return None, None, None


def format_reward(completions, **kwargs):
    """Calculate reward based on format compliance.
    
    Args:
        completions: List of model completion dictionaries
        
    Returns:
        List of reward scores (1.0 for valid format, 0.0 otherwise)
    """
    rewards = []
    for completion in completions:
        content = completion[0]["content"]
        points, labels, bbox = parse_custom_format(content)
        
        # Check for thinking content
        think_match = re.search(r"<think>([\s\S]*?)</think>", content)
        think_content = think_match.group(1).strip() if think_match else None
        
        format_reward = 1.0 if (points is not None and labels is not None and 
                               bbox is not None and think_content is not None) else 0.0
        rewards.append(format_reward)
    return rewards


def length_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        content = completion[0]["content"]
        points, _, _ = parse_custom_format(content)
        
        if points is not None and len(points) > 0:
            num_points = len(points)
            ideal = 8
            sigma = 2
            rewards.append(exp(-((num_points - ideal) ** 2) / (2 * sigma ** 2)))
        else:
            rewards.append(0)
    return rewards


def _prepare_data(pred: np.ndarray, gt: np.ndarray) -> tuple:
    """Normalize prediction and ground truth data formats.
    
    Args:
        pred: Prediction array
        gt: Ground truth array
        
    Returns:
        Tuple of normalized (prediction, ground_truth)
    """    
    pred = pred.astype(_TYPE)
    gt = gt.astype(_TYPE)
    gt = (gt > 128).astype(_TYPE)  # Ensure binarization
    return pred, gt


def segmentation_reward(completions: List[str], 
                       gt_mask: List[np.ndarray],
                       image: List[PILImage.Image],
                       
                       **kwargs) -> List[float]:
    rewards = []
    sam_config = kwargs.get("sam_config")[0]
    sam_wrapper = SAMWrapper(
            model_path=sam_config["model_path"],
            device=sam_config["device"]
        )
    sm_calculator = Smeasure(alpha=kwargs.get('s_measure_alpha', 0.5))
    for completion, gt_mask, img in zip(completions, gt_mask, image):
        content = completion[0]["content"]
        points, labels, bbox = parse_custom_format(content)

        iou_reward = 0.0
        sm_reward = 0.0
        # point_penalty = 0.0
        
        if points is not None and len(points) > 0 and bbox is not None and labels is not None:

            if not isinstance(img, PILImage.Image):
                img = PILImage.fromarray(img)
            

            final_mask = np.zeros(RESIZE_SIZE[::-1], dtype=bool)

            if (points is not None and labels is not None) or (bbox is not None):
                if not isinstance(img, PILImage.Image):
                    img = PILImage.fromarray(img)

                if bbox is not None and len(bbox.shape) == 2:  
                    for b in bbox:
                        b = b.tolist()
                        if points is not None and labels is not None:
                            in_bbox_mask = (
                                (points[:, 0] >= b[0]) & (points[:, 0] <= b[2]) &
                                (points[:, 1] >= b[1]) & (points[:, 1] <= b[3])
                            )
                            selected_points = points[in_bbox_mask]
                            selected_labels = labels[in_bbox_mask]
                        else:
                            selected_points, selected_labels = None, None

                        try:
                            mask, _ = sam_wrapper.predict(
                                img,
                                selected_points.tolist() if selected_points is not None and len(selected_points) > 0 else None,
                                selected_labels.tolist() if selected_labels is not None and len(selected_labels) > 0 else None,
                                b
                            )
                            final_mask |= (mask > 0)
                        except Exception as e:
                            print(f"Error in mask prediction for bbox {b}: {str(e)}")
                            continue

                    mask_pred = final_mask

                else:
                    try:
                        mask_pred, _ = sam_wrapper.predict(
                            img,
                            points.tolist() if points is not None else None,
                            labels.tolist() if labels is not None else None,
                            bbox.tolist() if bbox is not None else None
                        )
                        mask_pred = mask_pred > 0
                    except Exception as e:
                        print(f"Error in mask prediction: {str(e)}")
                        mask_pred = np.zeros(RESIZE_SIZE[::-1], dtype=bool)

            
            intersection = np.logical_and(mask_pred, gt_mask).sum()
            union = np.logical_or(mask_pred, gt_mask).sum()
            iou_reward = intersection / union if union > 0 else 0.0

            mask_pred = np.array(mask_pred).astype(_TYPE)
            gt_np = np.array(gt_mask).astype(_TYPE)
            mask_pred, gt_np = _prepare_data(mask_pred, gt_np)
            
            sm = sm_calculator.cal_sm(mask_pred, gt_np)
            sm_reward = max(0.0, min(1.0, sm))  
                

        total_reward = 0.7 * iou_reward + 0.3 * sm_reward #+  format_reward # + 0.1 * point_penalty
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
    messages = []
    SYSTEM_PROMPT = (
"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
"first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
"process should enclosed within <think> </think> tags, and the bounding box, points and points labels should be enclosed within <bbox></bbox>, <points></points>, and <labels></labels>, respectively. i.e., "
"<think> reasoning process here </think> <bbox>[x1,y1,x2,y2]</bbox>, <points>[[x1,y1],[x2,y2],...]</points>, <labels>[1,0,...]</labels>"
"Where 1 indicates a foreground (object) point, and 0 indicates a background point. Only generate a moderate number of points, enough to clearly outline the object without overloading the system."
    )
    messages.append({
        "role": "system",
        "content": [{"type": "text", "text": SYSTEM_PROMPT}],
    })
    messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": 
                        """
Identify and segment all the primary objects in the image.

Output the result using the exact format:
<think> reasoning process here </think> <bbox>[x1,y1,x2,y2]</bbox>, <points>[[x3,y3],[x4,y4],...]</points>, <labels>[1,0,...]</labels>
"""
                        # "Generate sparse prompt points for to generate a detailed mask for the salient object in this image. "
                        # "Output coordinates in <points>[[x1,y1],[x2,y2],...]</points> "
                        # "and labels in <labels>[1,0,...]</labels>, (0 for background, 1 for object). "
                        # "e.g. <points>[[100,200],[150,250],[151,252],[232,132],[2,2],[10,0]]</points> <labels>[1,0,1,1,0,0]</labels>"
                    }
                ],
            }
    )
    return {
        "prompt": messages,
    }


class SegR1Trainer(Qwen2VLGRPOVLLMTrainerModified):
    
    def __init__(self, *args, sam_config=None, **kwargs):
        super().__init__(*args, **kwargs)


# Reward functions registry
reward_funcs_registry = {
    "segment": segmentation_reward,
    "format": format_reward,
    "length": length_reward,
}


def main(script_args, training_args, model_args):
    # Initialize reward functions
    reward_funcs = ["segment", "format"]
    reward_funcs = [reward_funcs_registry[func] for func in reward_funcs]
    
    # Load dataset
    # dataset = load_dataset(
    #     "imagefolder", 
    #     data_dir=script_args.dataset_image,
    #     split='train',
    # )

    # # Add ground truth paths
    # def add_gt_paths(example):
    #     img_path = example["image"].filename
    #     file_name = os.path.basename(img_path)
    #     return {
    #         "gt_path": os.path.join(
    #             script_args.dataset_gt, 
    #             file_name.replace(".jpg", ".png")
    #         )
    #     }
    
    # dataset = dataset.map(add_gt_paths)
    # dataset = dataset.map(resize_image_and_mask)
    
    image_dirs = script_args.dataset_image.split(',')
    gt_dirs = script_args.dataset_gt.split(',')

    image_datasets = []
    for dir_path in image_dirs:
        image_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        ds = load_dataset("imagefolder", data_files={"train": image_files}, split='train')
        image_datasets.append(ds)
    dataset = concatenate_datasets(image_datasets) 
    # Optionally cap the number of samples used for training
    if script_args.max_train_samples is not None:
        max_n = int(script_args.max_train_samples)
        if max_n > 0:
            dataset = dataset.select(range(min(max_n, len(dataset))))

    def add_gt_paths(example):
        img_path = example["image"].filename
        file_name = os.path.basename(img_path)
        for gt_dir in gt_dirs:
            gt_path = os.path.join(gt_dir, file_name.replace(".jpg", ".png"))
            if os.path.exists(gt_path):
                return {"gt_path": gt_path}
        return {"gt_path": os.path.join(gt_dirs[0], file_name.replace(".jpg", ".png"))}
    dataset = dataset.map(add_gt_paths)
    dataset = dataset.map(resize_image_and_mask)
    dataset = dataset.map(
        lambda x: {
            "sam_config": {
                "device": script_args.sam_device,
                "model_path": script_args.sam_checkpoint
            }
        }
    )

    # Verify dataset structure
    if "image" not in dataset.features:
        raise ValueError("Dataset is missing required 'image' field")
    
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
        sam_config={
            "model_path": script_args.sam_checkpoint,
            "device": script_args.sam_device
        }
    )
    
    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)