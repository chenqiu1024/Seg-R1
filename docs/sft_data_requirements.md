### SFT input data: format and folder structure

This document describes the dataset format expected by the SFT stage (invoked via `bash scripts/sft.sh`). The SFT code loads a Hugging Face dataset by name and expects specific fields per example to build a vision-language chat sample.

### How SFT loads data

- The launch command runs:

```bash
accelerate launch --config_file seg-r1/configs/zero2.yaml \
  seg-r1/src/open_r1/sft.py \
  --config seg-r1/configs/qwen2vl_sft_config.yaml
```

- In `seg-r1/configs/qwen2vl_sft_config.yaml`:

```yaml
dataset_name: geshang/FCoT
dataset_configs:
- all
```

- In `sft.py` the dataset is loaded as:

```208:209:seg-r1/src/open_r1/sft.py
dataset = load_dataset(script_args.dataset_name, split="train")
```

That means the dataset must be published on Hugging Face Hub (or locally recognized by name) and provide a `train` split.

### Required fields in each dataset example

`sft.py` constructs a multi-turn chat with an image using these keys:

```123:141:seg-r1/src/open_r1/sft.py
thinking = example.get("thinking")
problem = example.get("problem")
solution = example.get("solution")
image = example.get("image")
messages.append({
  "role": "user",
  "content": [
    {"type": "text", "text": problem},
    {"type": "image", "image": image},
  ]
})
messages.append({
  "role": "assistant",
  "content": f"{thinking} {solution}",
})
```

Therefore, each example must include:

- `problem` (string): The user prompt/question.
- `image` (image field): The associated image content (HF Datasets image type or path that the dataset builder resolves to an image).
- `thinking` (string): The chain-of-thought segment wrapped later into the assistant message.
- `solution` (string): The final assistant content appended after `thinking`.

Optional:

- `system` (string): If present, becomes a system message; otherwise a default system prompt describing the expected `<think>`, `<bbox>`, `<points>`, and `<labels>` output format is inserted.

### Data modalities and processing

- The processor is created from the model (Qwen2.5-VL) and `collate_fn` applies the chat template to the messages and processes images using `process_vision_info`.
- Images are expected to be valid inputs for the model processor (e.g., PIL images, NumPy arrays, or paths supported by the dataset). No explicit image resizing or mask fields are required for SFT.

### Dataset structure examples

- Remote HF dataset (preferred): publish a dataset with a `train` split containing the required columns (`problem`, `image`, `thinking`, `solution`, optional `system`). For example, the default config references `geshang/FCoT` with config `all`.

- Local dataset by name: if you provide a local dataset name to `--dataset_name`, it must be loadable by `datasets.load_dataset(name, split="train")` and yield examples with the same fields. If using an `imagefolder`-style dataset, you would need a custom builder to produce the required text fields; plain image folders are insufficient for SFT because `thinking`, `problem`, and `solution` are required.

### Notes and pitfalls

- The SFT pipeline does not read masks or ground-truth segmentation maps; it only needs images and text fields to supervise the model to output the target prompt format.
- Ensure the `train` split exists; otherwise `load_dataset(..., split="train")` will fail.
- The `dataset_configs` entry in the YAML is not used directly by `sft.py` for loading but may be relevant if the dataset name expects a specific configuration when published on the Hub.


