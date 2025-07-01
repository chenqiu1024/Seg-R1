
import os
import json
import argparse
import io
import base64
import re
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.multiprocessing import Process, Manager, set_start_method
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as TF
import cv2

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

SAM_MODEL_PATH = "third_party/sam2/checkpoints/sam2.1_hiera_large.pt"
RESIZE_SIZE    = (1024, 1024)
# ===================================================================

class SAMWrapper:
    def __init__(self, model_path: str, device: str):
        sam_cfg    = "configs/sam2.1/sam2.1_hiera_l.yaml"
        sam_model  = build_sam2(sam_cfg, model_path).to(device)
        self.predictor = SAM2ImagePredictor(sam_model)
        self.device    = device

    def predict(self, img: Image.Image,
                points: Optional[List[Tuple[int,int]]],
                labels: Optional[List[int]],
                bbox:   Optional[List[int]]) -> Tuple[np.ndarray, float]:
        img_np = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        self.predictor.set_image(img_np)

        pts = np.array(points) if points is not None else None
        lbs = np.array(labels) if labels is not None else None
        box = np.array(bbox)   if bbox   is not None else None

        mask, score, _ = self.predictor.predict(
            point_coords  = pts,
            point_labels  = lbs,
            box           = box,
            multimask_output=False
        )
        return mask[0], float(score[0])

def resize_to_base64(path, size=RESIZE_SIZE):
    img = Image.open(path).resize(size)
    buf = io.BytesIO(); img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()

def prepare_message(img_path: str, expr: str):
    b64 = resize_to_base64(img_path)
    sys = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process should enclosed within <think> </think> tags, and the bounding box, points and points labels should be enclosed within <bbox></bbox>, <points></points>, and <labels></labels>, respectively. i.e., "
    "<think> reasoning process here </think> <bbox>[x1,y1,x2,y2]</bbox>, <points>[[x3,y3],[x4,y4],...]</points>, <labels>[1,0,...]</labels>"
    "Where 1 indicates a foreground (object) point, and 0 indicates a background point."
    )
    return [
        {"role":"system", "content":[{"type":"text","text":sys}]},
        {"role":"user",   "content":[
            {"type":"image","image":f"data:image/jpeg;base64,{b64}"},
            {"type":"text", "text":  expr[0]}
        ]}
    ]

def parse_custom_format(content: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    point_pattern = r"<points>\s*(\[\s*(?:\[\s*\d+\s*,\s*\d+\s*\]\s*,?\s*)+\])\s*</points>"
    label_pattern = r"<labels>\s*(\[\s*(?:\d+\s*,?\s*)+\])\s*</labels>"
    bbox_pattern = r"<bbox>(\s*\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\])"
    
    point_match = re.search(point_pattern, content)
    label_match = re.search(label_pattern, content)
    bbox_match = re.search(bbox_pattern, content)
    
    def safe_convert_to_int_array(data: list) -> Optional[np.ndarray]:
        try:
            def convert(item):
                if isinstance(item, list):
                    return [convert(subitem) for subitem in item]
                else:
                    return int(item) if not isinstance(item, int) else item
                    
            converted = convert(data)
            return np.array(converted, dtype=np.int32)
        except Exception as e:
            print(f"Transform Error!: {e}")
            return None

    try:
        points = None
        if point_match:
            points = safe_convert_to_int_array(eval(point_match.group(1)))
        
        labels = None
        if label_match:
            labels = safe_convert_to_int_array(eval(label_match.group(1)))
        
        bbox = None
        if bbox_match:
            bbox_list = eval(bbox_match.group(1))
            if isinstance(bbox_list, list):
                try:
                    bbox = np.array([int(x) for x in bbox_list], dtype=np.int32)
                except:
                    bbox = None
        
        return points, labels, bbox
    except Exception as e:
        print(f"Parse Error!: {e}")
        return None, None, None

def calculate_ciou(pred: np.ndarray, gt: np.ndarray):
    i = np.logical_and(pred, gt).sum()
    u = np.logical_or(pred, gt).sum()
    return i/u if u>0 else 0.0


def init_model_on_gpu(model_path: str, gpu_id: int):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=f"cuda:{gpu_id}"
    ).eval()
    proc  = AutoProcessor.from_pretrained(model_path, use_fast=True)
    if hasattr(proc, "tokenizer"):
        proc.tokenizer.padding_side = "left"
    else:
        proc.padding_side = "left"
    return model, proc

def answer_question(batch_msgs, model, proc):
    texts = [proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
             for m in batch_msgs]
    imgs, vids = process_vision_info(batch_msgs)
    inp = proc(text=texts, images=imgs, videos=vids,
               padding=True, return_tensors="pt").to(model.device)
    gen = model.generate(**inp, use_cache=True, max_new_tokens=512)
    trims = [g[len(i):] for i,g in zip(inp.input_ids, gen)]
    return proc.batch_decode(trims, skip_special_tokens=True)

def infer_on_single_gpu(model_path, gpu_id, samples, batch_size, return_dict):
    torch.cuda.set_device(gpu_id)
    model, proc = init_model_on_gpu(model_path, gpu_id)
    sam_wrapper = SAMWrapper(SAM_MODEL_PATH, device=f"cuda:{gpu_id}")

    all_scores = []
    chunks = [samples[i:i+batch_size] for i in range(0, len(samples), batch_size)]

    for chunk in tqdm(chunks, desc=f"GPU{gpu_id}", position=gpu_id):
        msgs = [prepare_message(s["img"], s["expr"]) for s in chunk]
        outs = answer_question(msgs, model, proc)

        for out, s in zip(outs, chunk):
            pts, lbs, bb = parse_custom_format(out)
            img = Image.open(s["img"]).convert("RGB")
            img = TF.resize(img, RESIZE_SIZE)

            if (pts is not None and lbs is not None and len(pts) == len(lbs)) or bb is not None:
                if pts is not None and lbs is not None and len(pts) == len(lbs):
                    pm, _ = sam_wrapper.predict(img, pts.tolist(), lbs.tolist(), bb)
                else:
                    pm, _ = sam_wrapper.predict(img, None, None, bb.tolist())
                pm = pm > 0
            else:
                pm = np.zeros(RESIZE_SIZE[::-1], dtype=bool)

            # resize to GT size
            gt = s["gt"][0].astype(bool)
            orig_h, orig_w = gt.shape
            pm = Image.fromarray(pm.astype(np.uint8) * 255)
            pm = TF.resize(pm, (orig_h, orig_w))
            pm = np.array(pm) > 128

            intersect = np.logical_and(pm, gt).sum()
            union = np.logical_or(pm, gt).sum()
            iou = intersect / (union + 1e-10)

            all_scores.append({'intersect': intersect, 'union': union, 'iou': iou})

    return_dict[gpu_id] = all_scores


def multi_gpu_evaluate(samples, model_path, gpu_ids, batch_size):
    set_start_method("spawn", force=True)
    mgr = Manager()
    ret = mgr.dict()
    p_list = []
    ids = [int(x) for x in gpu_ids.split(",")]

    chunk = len(samples) // len(ids)
    for i, g in enumerate(ids):
        sub = samples[i*chunk: len(samples) if i == len(ids)-1 else (i+1)*chunk]
        p = Process(target=infer_on_single_gpu, args=(model_path, g, sub, batch_size, ret))
        p.start()
        p_list.append(p)

    for p in p_list:
        p.join()

    scores = []
    for g in ids:
        scores.extend(ret[g])


    total_intersect = sum(s['intersect'] for s in scores)
    total_union = sum(s['union'] for s in scores)
    total_iou_sum = sum(s['iou'] for s in scores)
    total_count = len(scores)
    print(total_count)

    ciou = total_intersect / (total_union + 1e-10)
    giou = total_iou_sum / total_count


    return ciou, giou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",   required=True)
    # parser.add_argument("--image_root",  required=True)
    parser.add_argument(
        "--split", type=str, default="test",
        help="Dataset splits to evaluate, comma-separated (e.g., test,testA,testB)"
    )
    parser.add_argument("--output_file", default="results.json")
    parser.add_argument("--model_path",  required=True)
    parser.add_argument("--gpu_ids",     default="0,1,2,3")
    parser.add_argument("--batch_size",  type=int, default=8)
    args = parser.parse_args()
    args.split = ["test","val"] if args.split=="all" else args.split.split(",")

    
    results = {}
    for split in args.split:
        dataset_name = f"ReasonSeg|{split}"
        print(f"\n{'='*50}")
        print(f"Evaluating {dataset_name.upper()} dataset...")
        print(f"  Split: {split}")
        dataset_results = {}

        from eval.datasets.ReasonSeg_Dataset import ReasonSegDataset
        dataset = ReasonSegDataset(
            base_image_dir=args.data_root,
            samples_per_epoch=80000,
            precision="fp32",
            image_size=RESIZE_SIZE[0],
            num_classes_per_sample=1,
            exclude_val=False,
            reason_seg_data=dataset_name, 
            explanatory=-1,
        )
        dataset._set_len(len(dataset.reason_seg_data[0]))

        samples = []
        for idx in range(len(dataset)):
            item = dataset[idx]
            img_path = item[0]
            gt_mask  = item[1].numpy().astype(bool)       # H×W bool
            expr     = item[2]                            
            samples.append({"img": img_path, "gt": gt_mask, "expr": expr})

        
        ciou, giou = multi_gpu_evaluate(
            samples,
            args.model_path,
            args.gpu_ids,
            args.batch_size
        )

        out = {"cIoU": ciou, "gIoU": giou}
        dataset_results[split] = out
        print(f"Done! cIoU = {ciou:.4f}, gIoU = {giou:.4f}\n")
        print(f"=============================================")
        results[dataset_name] = dataset_results
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.output_file}")

if __name__ == "__main__":
    main()
