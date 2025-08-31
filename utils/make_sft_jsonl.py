#!/usr/bin/env python3

"""
Generate an SFT-ready JSONL from an image folder.

Each line contains required fields for Seg-R1 SFT stage:
- problem: instruction text
- image: path to image (loader resolves via HF datasets)
- thinking: placeholder chain-of-thought (optionally empty)
- solution: final short answer text

Usage
  python utils/make_sft_jsonl.py \
    --image_dir datasets/seg_r1_md/Task09_Spleen/size512/canonical/images \
    --output_jsonl datasets/seg_r1_md/Task09_Spleen/sft/train.jsonl \
    --problem_template "Segment the salient lesion." \
    --solution_template "<think>...</think> <bbox>[]</bbox> <points>[]</points> <labels>[]</labels>"

Notes
- Only images are required for SFT; masks are not used.
- You may later publish this as a HuggingFace dataset with a train split.
"""

import argparse
import json
import os
from typing import List


def list_images(d: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return [os.path.join(d, f) for f in sorted(os.listdir(d)) if os.path.splitext(f)[1].lower() in exts]


def main() -> None:
    ap = argparse.ArgumentParser(description="Create SFT JSONL from an image directory")
    ap.add_argument("--image_dir", required=True, help="Directory containing images")
    ap.add_argument("--output_jsonl", required=True, help="Output JSONL path")
    ap.add_argument("--problem_template", default="Describe and segment the salient object.", help="Problem text for each sample")
    ap.add_argument("--thinking_template", default="<think>Consider the object boundary and context.</think>", help="Chain-of-thought placeholder")
    ap.add_argument("--solution_template", default="Return points/labels following the required format.", help="Solution text appended after thinking")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
    images = list_images(args.image_dir)

    n = 0
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for img in images:
            ex = {
                "problem": args.problem_template,
                "image": img,
                "thinking": args.thinking_template,
                "solution": args.solution_template,
            }
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote {n} examples to {args.output_jsonl}")


if __name__ == "__main__":
    main()


