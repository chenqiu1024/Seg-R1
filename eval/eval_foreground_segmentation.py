from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import torch
import json
import tqdm
from math_verify import parse, verify
import argparse
import pandas as pd
from torch.multiprocessing import Process, set_start_method, Manager
from transformers.utils.logging import disable_progress_bar
disable_progress_bar()
import numpy as np
from PIL import Image as PILImage
import torchvision.transforms.functional as TF
import base64
import io
from PIL import Image
import cv2
import os
import re
from typing import List, Tuple, Optional
import torchvision.transforms.functional as TF
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from utils.evaluator import SegmentationEvaluator

RESIZE_SIZE = (768, 768)
class SAMWrapper:
    def __init__(self, model_path: str, device: str = "cuda:3"):
        checkpoint = model_path
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        sam_model = build_sam2(model_cfg, checkpoint)
        self.device = torch.device(device)
        
        # 移动到指定 GPU
        sam_model = sam_model.to(self.device)
        
        # 初始化 predictor
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
            multimask_output=False,
        )
        
        self.last_mask = mask_pred[0]
        return mask_pred[0], score[0]
sam_wrapper = SAMWrapper(model_path="third_party/sam2/checkpoints/sam2.1_hiera_large.pt", device="cuda:3")
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>> 1. get evaluation configuration <<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def get_eval_config():
    parser = argparse.ArgumentParser(description="Inference script for GeoQA evaluation.")
    parser.add_argument("--model_path", required=True, type=str, help="Path to the model checkpoint (e.g., qwen2vl model or a fine-tuned model).")
    parser.add_argument("--dataset", required=True, type=str, help="Test dataset(s), separated by +")
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size for inference. Reduce if GPU OOM (default: 50).")
    parser.add_argument("--output_path", default="eval/log", type=str, help="Path to save inference result (e.g., JSON file).")
    parser.add_argument("--prompt_path", required=True, type=str, help="Path to the prompts JSONL file")
    all_gpu = ",".join(map(str, range(torch.cuda.device_count())))
    parser.add_argument("--gpu_ids", default=all_gpu, help="comma-separated list of GPU IDs to use")
    parser.add_argument('--vis_output_path', type=str, default=None, help="Path to save visualized predictions")

    args = parser.parse_args()
    return args

def resize_image_to_base64(image_path, size=(768, 768)):
    img = Image.open(image_path)
    img = img.resize(size) 
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")  
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>> 2. load testset <<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def prepare_test_messages(testset_path):
    testset_data = pd.read_json(testset_path, lines=True).to_dict(orient="records")
    # QUESTION_TEMPLATE = "{Question} Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."
    tested_messages = []
    for i in testset_data:
        img_base64 = resize_image_to_base64(i['image_path'])
        messages = []
        SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process should enclosed within <think> </think> tags, and the bounding box, points and points labels should be enclosed within <bbox></bbox>, <points></points>, and <labels></labels>, respectively. i.e., "
    "<think> reasoning process here </think> <bbox>[x1,y1,x2,y2]</bbox>, <points>[[x3,y3],[x4,y4],...]</points>, <labels>[1,0,...]</labels>"
    "Where 1 indicates a foreground (object) point, and 0 indicates a background point."

        )
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        })
        messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"data:image/jpeg;base64,{img_base64}"},
                        {"type": "text", "text": 
                            """
Identify and segment all the primary objects in the image.

Output the result using the exact format:
<think> reasoning process here </think> <bbox>[x1,y1,x2,y2]</bbox>, <points>[[x3,y3],[x4,y4],...]</points>, <labels>[1,0,...]</labels>
    """
                        }
                    ],
                }
        )

        tested_messages.append(messages)
    return testset_data, tested_messages




# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>> 3. use several GPUs to accelerate inference at testset <<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def init_model(model_path, gpu_id):
    """init a model(args.model_path) on a specific gpu"""
    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=f"cuda:{gpu_id}",
    )

    # default processer
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
    return model, processor

def answer_a_batch_question_qwen(batch_messages, model, processor):
    """ let qwen answer a batch of questions """
    text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]        
    image_inputs, video_inputs = process_vision_info(batch_messages)
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    
    generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=1024) # do_sample=False
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    batch_output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return batch_output_text

def infer_on_single_gpu(model_path, device_id, chunk_of_tested_messages, batch_size, results=None):
    """init model on this single gpu and let it answer asign chunk of questions"""
    model, processor = init_model(model_path, device_id)
    
    ### split batch
    responses = []
    batch_messages_list = [chunk_of_tested_messages[start: start + batch_size] 
               for start in range(0, len(chunk_of_tested_messages), batch_size)]

    for batch_messages in tqdm.auto.tqdm(batch_messages_list, desc=f"GPU {device_id} progress", position=device_id, leave=False):
        batch_output_text = answer_a_batch_question_qwen(batch_messages, model, processor)
        
        responses.extend(batch_output_text)
    
    results[device_id] = responses
    return
        
        
def multi_gpu_inference(prompts, gpu_ids, model_path, batch_size):
    """ let each gpu (along with a model) answer a chunk of questions """
    set_start_method("spawn", force=True)
    manager = Manager()
    gpu_id2result = manager.dict()

    gpu_ids = [int(gpu_id.strip()) for gpu_id in gpu_ids.split(',')]
    num_gpus = len(gpu_ids)

    chunk_size = len(prompts) // num_gpus
    processes = []
    for i, gpu_id in enumerate(gpu_ids):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i != num_gpus - 1 else len(prompts)
        chunk = prompts[start_idx: end_idx]
        process = Process(target=infer_on_single_gpu, args=(model_path, gpu_id, chunk, batch_size, gpu_id2result))
        process.start()
        processes.append(process)

    # for process in tqdm.auto.tqdm(processes, desc="Inference progress", position=num_gpus, leave=True):
    for process in processes:
        process.join()

    all_predicts = []
    for gpu_id in gpu_ids:
        all_predicts.extend(gpu_id2result[gpu_id])

    return all_predicts
import re
import numpy as np
from typing import Tuple, Optional

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

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>> 4. compute metrics <<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def compute_metrics(testset_data, all_predicts, vis_output_path=None):
    final_output = []
    correct_number = 0

    for input_example, model_output in tqdm.tqdm(zip(testset_data, all_predicts), total=len(testset_data), desc="Processing"):
        predict_path = vis_output_path
        original_output = model_output
        ground_truth = input_example['ground_truth']
        mask = PILImage.open(ground_truth).convert("L")
        mask_resized = TF.resize(mask, RESIZE_SIZE)
        
        gt_mask = np.array(mask_resized) > 127 

        points, labels, bbox = parse_custom_format(original_output)
        # print(f"Parsed points: {points}, labels: {labels}, bbox: {bbox}")
        img = PILImage.open(input_example['image_path']).convert("RGB")
        img_size = img.size
        img = TF.resize(img, RESIZE_SIZE)        
            
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

            
            pred_mask = PILImage.fromarray((mask_pred * 255).astype(np.uint8))
            pred_mask = TF.resize(pred_mask, (img_size[1], img_size[0]), interpolation=TF.InterpolationMode.BILINEAR)
            pred_mask = pred_mask.convert("L")
            pred_mask_path = f"{predict_path}/{os.path.basename(input_example['ground_truth'])}"
            pred_mask.save(pred_mask_path)



if __name__ == "__main__":
    args = get_eval_config()
    if args.vis_output_path is not None:
        os.makedirs(args.vis_output_path, exist_ok=True)
    testset_data, tested_messages = prepare_test_messages(testset_path=args.prompt_path)
    gt_root = os.path.dirname(testset_data[0]["ground_truth"])
    all_predicts = multi_gpu_inference(tested_messages, args.gpu_ids, args.model_path, args.batch_size)
    compute_metrics(testset_data, all_predicts, args.vis_output_path)
    evaluator = SegmentationEvaluator(
        gt_root=gt_root,
        pred_root=args.vis_output_path,
        model_lst=["Seg-R1"],
        save_dir=args.output_path,
        data_lst=args.dataset,
        check_integrity=True,
    )
    evaluator.run()

