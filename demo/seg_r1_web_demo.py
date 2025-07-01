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


MODEL_PATH = "geshang/Seg-R1-7B" 
DEVICE_QWEN = "cuda:0"
DEVICE_SAM = "cuda:0"
RESIZE_SIZE = (1024, 1024)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=DEVICE_QWEN,
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

def prepare_test_messages(image, prompt):
    buffered = io.BytesIO()
    image = TF.resize(image, RESIZE_SIZE)
    image.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    if "segment" in prompt or "mask" in prompt:

        SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process should enclosed within <think> </think> tags, and the bounding box, points and points labels should be enclosed within <bbox></bbox>, <points></points>, and <labels></labels>, respectively. i.e., "
    "<think> reasoning process here </think> <bbox>[x1,y1,x2,y2]</bbox>, <points>[[x3,y3],[x4,y4],...]</points>, <labels>[1,0,...]</labels>"
    "Where 1 indicates a foreground (object) point, and 0 indicates a background point."

        )
    else:
        SYSTEM_PROMPT = (
    "You're a helpful visual assistant."

        )

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"data:image/jpeg;base64,{img_base64}"},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    return [messages]

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

def run_pipeline(image: PILImage.Image, prompt: str):
    img_original = image.copy()
    img_resized = TF.resize(image, RESIZE_SIZE)

    messages = prepare_test_messages(img_resized, prompt)
    output_text = answer_question(messages)[0]

    points, labels, bbox = parse_custom_format(output_text)
    print(f"Output text: {output_text}")
    print(f"Parsed points: {points}, labels: {labels}, bbox: {bbox}")

    # if points is None or labels is None or bbox is None:
    #     return output_text, None
    img = img_resized
    mask_pred = None
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
    else:
        return output_text, None
    mask_np = mask_pred
    mask_img = PILImage.fromarray((mask_np * 255).astype(np.uint8)).resize(img_original.size)
    mask_img = mask_img.convert("L")
    resized_w, resized_h = img_resized.size
    original_w, original_h = img_original.size

    scale_x = original_w / resized_w
    scale_y = original_h / resized_h
    visualized_img = visualize_masks_on_image_v2(
        image, 
        masks_np=[mask_np],
        alpha=0.6
    )
    return output_text, visualized_img

gr.Interface(
    fn=run_pipeline,
    inputs=[
        gr.Image(type="pil", label="Image"),
        gr.Textbox(lines=2, label="Text")
    ],
    outputs=[
        gr.Textbox(label="Model Output"),
        gr.Image(type="pil", label="Mask Prediction")
    ],
    title="Seg-R1",
).launch(share=True)
