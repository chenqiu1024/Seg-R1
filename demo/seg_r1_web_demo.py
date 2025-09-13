import gradio as gr
from PIL import Image as PILImage
import torchvision.transforms.functional as TF
import numpy as np
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import re
import io
import base64
import cv2
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path
import random

CACHE_DIR = "/root/autodl-tmp/models" 
MODEL_PATH = "geshang/Seg-R1-7B" 
DEVICE_QWEN = "cuda:0"
DEVICE_SAM = "cuda:0"
RESIZE_SIZE = (1024, 1024)
EPSILON_DEFAULT = 0.1

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=DEVICE_QWEN,
    cache_dir=CACHE_DIR,
)
processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)

# SAM Wrapper
class SAMWrapper:
    def __init__(self, model_path: str, device: str = "cuda:1"):
        checkpoint = model_path
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        sam_model = build_sam2(model_cfg, checkpoint)
        self.device = torch.device(device)
        
        sam_model = sam_model.to(self.device)
        
        self.predictor = SAM2ImagePredictor(sam_model)
        self.last_mask = None
        
    def predict(self, image: PILImage.Image, 
               points: List[Tuple[int, int]], 
               labels: List[int],
               bbox: Optional[List[List[int]]] = None) -> Tuple[np.ndarray, float]:
               

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

sam_wrapper = SAMWrapper("third_party/sam2/checkpoints/sam2.1_hiera_large.pt", device=DEVICE_SAM)


def parse_custom_format(content: str):

    point_pattern = r"<points>\s*(\[\s*(?:\[\s*\d+\s*,\s*\d+\s*\]\s*,?\s*)+\])\s*</points>"
    label_pattern = r"<labels>\s*(\[\s*(?:\d+\s*,?\s*)+\])\s*</labels>"
    bbox_pattern  = r"<bbox>\s*(\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\])\s*</bbox>"

    point_matches = re.findall(point_pattern, content)
    label_matches = re.findall(label_pattern, content)
    bbox_matches = re.findall(bbox_pattern, content)

    try:
        # Collect all points blocks
        points_list = []
        for pm in point_matches:
            arr = np.array(eval(pm))
            if len(arr.shape) == 2 and arr.shape[1] == 2:
                points_list.append(arr)
        points = np.concatenate(points_list, axis=0) if points_list else None

        # Collect all labels blocks
        labels_list = []
        for lm in label_matches:
            arr = np.array(eval(lm)).reshape(-1)
            labels_list.append(arr)
        labels = np.concatenate(labels_list, axis=0) if labels_list else None

        # Validate lengths if both present
        if points is not None and labels is not None:
            if len(labels) != points.shape[0]:
                labels = None

        # BBoxes (already multiple)
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

def prepare_test_messages(image, prompt, ratio: float = None, epsilon: float = EPSILON_DEFAULT):
    buffered = io.BytesIO()
    image = TF.resize(image, RESIZE_SIZE)
    image.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    if "segment" in prompt or "mask" in prompt:
        SYSTEM_PROMPT_ORIG = (
          "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
          "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
          "process should enclosed within <think> </think> tags, and the bounding box, points and points labels should be enclosed within <bbox></bbox>, <points></points>, and <labels></labels>, respectively. i.e., "
          "<think> reasoning process here </think> <bbox>[x1,y1,x2,y2]</bbox>, <points>[[x3,y3],[x4,y4],...]</points>, <labels>[1,0,...]</labels>"
          "Where 1 indicates a foreground (object) point, and 0 indicates a background point."
        )
        safe_ratio = max(1e-6, min(1.0, float(ratio)))
        # Choose concrete counts: random bboxes in [1,20], points derived from ratio and clamped to [1,20]
        num_bboxes = random.randint(1, 10)
        num_points = max(1, min(20, int(round(safe_ratio * num_bboxes))))
        SYSTEM_PROMPT_RATIO = (
            "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
            "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
            "process should enclosed within <think> </think> tags, and the bounding box, points and points labels should be enclosed within <bbox></bbox>, <points></points>, and <labels></labels>, respectively. i.e., "
            "<think> reasoning process here </think> <bbox>[x1,y1,x2,y2]</bbox>, <points>[[x3,y3],[x4,y4],...]</points>, <labels>[1,0,...]</labels>. "
            "There could be multiple <bbox> blocks. "
            "Constraints: "
            f"Generate EXACTLY {num_bboxes} separate <bbox> blocks. "
            f"Generate EXACTLY ONE <points> block containing EXACTLY {num_points} coordinate pairs. "
            f"Generate EXACTLY ONE <labels> block containing EXACTLY {num_points} labels (1 or 0), matching the points order. "
            "Do NOT include any extra text outside these tags."
        )
    else:
        SYSTEM_PROMPT_ORIG = (
            "You're a helpful visual assistant."
        )
        SYSTEM_PROMPT_RATIO = (
            "You're a helpful visual assistant."
        )

    messages_orig = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_ORIG}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"data:image/jpeg;base64,{img_base64}"},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    messages_ratio = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_RATIO}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"data:image/jpeg;base64,{img_base64}"},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    return [messages_orig], [messages_ratio]

def answer_question(batch_messages):
    text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
    image_inputs, video_inputs = process_vision_info(batch_messages)
    inputs = processor(text=text, images=image_inputs, videos=video_inputs, return_tensors="pt", padding=True).to(model.device)
    outputs = model.generate(**inputs, use_cache=True, max_new_tokens=1024)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, outputs)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def visualize_masks_on_image_v2(
    image: PILImage.Image,
    masks_np: list,  
    colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255),  
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 128, 255)],
    alpha=0.5,  
):
    image_np = np.array(image)
    color_mask = np.zeros((image_np.shape[0], image_np.shape[1], 3), dtype=np.uint8)
    
    mask = masks_np[0]
    mask = mask.astype(np.uint8)
    if mask.shape[:2] != image_np.shape[:2]:
        mask = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]))
    
    color = colors[0]
    
    color_mask[:, :, 0] = color_mask[:, :, 0] | (mask * color[0])
    color_mask[:, :, 1] = color_mask[:, :, 1] | (mask * color[1])
    color_mask[:, :, 2] = color_mask[:, :, 2] | (mask * color[2])
    
    blended = cv2.addWeighted(image_np, 1 - alpha, color_mask, alpha, 0)
    
    blended_pil = Image.fromarray(blended)
    
    edge_layer = Image.new("RGBA", blended_pil.size, (0, 0, 0, 0))
    if mask.shape[:2] != blended_pil.size[::-1]:
        mask = cv2.resize(mask, blended_pil.size)
    
    mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
    edge = mask_pil.filter(ImageFilter.FIND_EDGES)
    edge = edge.point(lambda p: 255 if p > 10 else 0)

    blended_pil = blended_pil.convert("RGBA")
    blended_pil = Image.alpha_composite(blended_pil, edge_layer)
    
    
    return blended_pil.convert("RGB")

def visualize_annotations_on_image(
    image: PILImage.Image,
    points=None,
    labels=None,
    bboxes=None,
):
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    width, height = img_draw.size
    scale_x = width / RESIZE_SIZE[0]
    scale_y = height / RESIZE_SIZE[1]

    if bboxes is not None:
        if isinstance(bboxes, np.ndarray):
            if len(bboxes.shape) == 1 and len(bboxes) == 4:
                b_iter = [bboxes.tolist()]
            else:
                b_iter = bboxes.tolist()
        else:
            b_iter = bboxes
        for b in b_iter:
            x1 = int(b[0] * scale_x)
            y1 = int(b[1] * scale_y)
            x2 = int(b[2] * scale_x)
            y2 = int(b[3] * scale_y)
            draw.rectangle([x1, y1, x2, y2], outline=(255, 215, 0), width=4)

    if points is not None:
        pts = points.tolist() if isinstance(points, np.ndarray) else points
        lbls = labels.tolist() if (labels is not None and isinstance(labels, np.ndarray)) else labels
        for idx, (x, y) in enumerate(pts):
            xi = int(x * scale_x)
            yi = int(y * scale_y)
            r = 6
            is_fg = False
            if lbls is not None:
                try:
                    is_fg = int(lbls[idx]) == 1
                except Exception:
                    is_fg = False
            color = (0, 255, 0) if is_fg else (255, 0, 0)
            draw.ellipse([xi - r, yi - r, xi + r, yi + r], fill=color, outline=(0, 0, 0))

    return img_draw

def run_pipeline(image: PILImage.Image, prompt: str, ratio: float):
    img_original = image.copy()
    img_resized = TF.resize(image, RESIZE_SIZE)

    # Prepare original and ratio-enforced messages (batched for one forward)
    messages_orig, messages_ratio = prepare_test_messages(img_resized, prompt, ratio=ratio, epsilon=EPSILON_DEFAULT)
    outputs_text = answer_question(messages_orig + messages_ratio)
    output_text_orig = outputs_text[0] if len(outputs_text) > 0 else ""
    output_text_ratio = outputs_text[1] if len(outputs_text) > 1 else ""
    # outputs_text = answer_question(messages_orig)[0] ## For debug
    # output_text_orig = outputs_text ## For debug
    # output_text_ratio = outputs_text ## For debug

    points_orig, labels_orig, bbox_orig = parse_custom_format(output_text_orig)
    print(f"[ORIG] Output text: {output_text_orig}")
    print(f"[ORIG] Parsed points: {points_orig}, labels: {labels_orig}, bbox: {bbox_orig}")

    points_ratio, labels_ratio, bbox_ratio = parse_custom_format(output_text_ratio)
    print(f"[RATIO] Output text: {output_text_ratio}")
    print(f"[RATIO] Parsed points: {points_ratio}, labels: {labels_ratio}, bbox: {bbox_ratio}")

    # if points is None or labels is None or bbox is None:
    #     return output_text, None
    img = img_resized
    def compute_visualization(points, labels, bbox):
        local_img = img
        mask_pred_local = None
        final_mask_local = np.zeros(RESIZE_SIZE[::-1], dtype=bool)

        if (points is not None and labels is not None) or (bbox is not None):
            if not isinstance(local_img, PILImage.Image):
                local_img = PILImage.fromarray(local_img)

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
                            local_img,
                            selected_points.tolist() if selected_points is not None and len(selected_points) > 0 else None,
                            selected_labels.tolist() if selected_labels is not None and len(selected_labels) > 0 else None,
                            b
                        )
                        final_mask_local |= (mask > 0)
                    except Exception as e:
                        print(f"Error in mask prediction for bbox {b}: {str(e)}")
                        continue

                mask_pred_local = final_mask_local

            else:
                try:
                    mask_pred_local, _ = sam_wrapper.predict(
                        local_img,
                        points.tolist() if points is not None else None,
                        labels.tolist() if labels is not None else None,
                        bbox.tolist() if bbox is not None else None
                    )
                    mask_pred_local = mask_pred_local > 0
                except Exception as e:
                    print(f"Error in mask prediction: {str(e)}")
                    mask_pred_local = np.zeros(RESIZE_SIZE[::-1], dtype=bool)
        else:
            return None

        mask_np_local = mask_pred_local
        vis_img_local = visualize_masks_on_image_v2(
            image,
            masks_np=[mask_np_local],
            alpha=0.6
        )
        return vis_img_local

    vis_ann_orig = None
    if (points_orig is not None and len(points_orig) > 0) or (bbox_orig is not None and len(bbox_orig) > 0):
        vis_ann_orig = visualize_annotations_on_image(image, points_orig, labels_orig, bbox_orig)

    vis_ann_ratio = None
    if (points_ratio is not None and len(points_ratio) > 0) or (bbox_ratio is not None and len(bbox_ratio) > 0):
        vis_ann_ratio = visualize_annotations_on_image(image, points_ratio, labels_ratio, bbox_ratio)

    visualized_img_orig = vis_ann_orig if vis_ann_orig is not None else compute_visualization(points_orig, labels_orig, bbox_orig)
    visualized_img_ratio = vis_ann_ratio if vis_ann_ratio is not None else compute_visualization(points_ratio, labels_ratio, bbox_ratio)

    return output_text_orig, visualized_img_orig, messages_ratio[0][0], output_text_ratio, visualized_img_ratio

gr.Interface(
    fn=run_pipeline,
    inputs=[
        gr.Image(type="pil", label="Image"),
        gr.Textbox(lines=2, label="Text"),
        gr.Slider(0.0, 10.0, step=0.01, value=1.0, label="Ratio")
    ],
    outputs=[
        gr.Textbox(label="Model Output (Original)"),
        gr.Image(type="pil", label="Mask Prediction (Original)"),
        gr.Textbox(label="Model System Prompt (Ratio)"),
        gr.Textbox(label="Model Output (Ratio)"),
        gr.Image(type="pil", label="Mask Prediction (Ratio)")
    ],
    title="Seg-R1",
).launch(share=True)
