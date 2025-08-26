### Fine-tune on SOD: input data format and folder structure

This document describes the dataset layout and file format required by the "Fine-tune on SOD" process (invoked via `bash scripts/run_grpo_sod.sh`). Under the hood it uses the same RL pipeline (`seg-r1/src/open_r1/grpo.py`) as general RL training, with a single SOD dataset.

### What the script passes to training

- **images root**: `--dataset_image`
- **masks root**: `--dataset_gt`

From the default script:

```bash
--dataset_image datasets/DUTS/DUTS-TR/DUTS-TR-Image \
--dataset_gt    datasets/DUTS/DUTS-TR/DUTS-TR-Mask
```

### Required directory layout

Two leaf directories are required for the chosen SOD dataset (e.g., DUTS train split):

```
datasets/DUTS/DUTS-TR/
├── DUTS-TR-Image/   # passed as --dataset_image
└── DUTS-TR-Mask/    # passed as --dataset_gt
```

Only the two leaf directories matter for the code; they can be located anywhere as long as you pass their paths.

### File naming and 1:1 pairing requirement

- Every image file in `--dataset_image` must have a corresponding mask file in `--dataset_gt`.
- The match is by basename with a fixed extension mapping `.jpg → .png`.
  - Example: `DUTS-TR-Image/ILSVRC2012_test_00000123.jpg` → `DUTS-TR-Mask/ILSVRC2012_test_00000123.png`
  - If your images are not `.jpg`, masks will not be found because the code replaces `.jpg` with `.png`. Rename images to `.jpg`.
  - Non-recursive scan: the image directory is scanned with `os.listdir(...)` (non-recursive). Put all images directly inside the specified `--dataset_image` folder.

### Accepted formats and preprocessing

- **Images**: Loaded via Hugging Face Datasets `imagefolder`, converted to RGB, and resized to 768×768. Aspect ratio is not preserved.
- **Masks**: Grayscale `.png`. Loaded, resized to 768×768, then binarized with threshold `> 127`.
  - Foreground: values > 127
  - Background: values ≤ 127

### Minimal example

```
datasets/DUTS/DUTS-TR/
├── DUTS-TR-Image/
│   ├── 0001.jpg
│   └── 0002.jpg
└── DUTS-TR-Mask/
    ├── 0001.png
    └── 0002.png
```

Run:

```bash
bash scripts/run_grpo_sod.sh \
  --dataset_image datasets/DUTS/DUTS-TR/DUTS-TR-Image \
  --dataset_gt    datasets/DUTS/DUTS-TR/DUTS-TR-Mask
```

### How the paths are used in code

The SOD run uses the RL loader in `grpo.py`. For a single dataset, the logic simplifies to:

```434:454:seg-r1/src/open_r1/grpo.py
image_dirs = script_args.dataset_image.split(',')
gt_dirs = script_args.dataset_gt.split(',')

image_datasets = []
for dir_path in image_dirs:
    image_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    ds = load_dataset("imagefolder", data_files={"train": image_files}, split='train')
    image_datasets.append(ds)
dataset = concatenate_datasets(image_datasets)

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
```

Mask loading and binarization (excerpt):

```42:51:seg-r1/src/open_r1/grpo.py
mask = PILImage.open(example["gt_path"]).convert("L")
mask_resized = TF.resize(mask, RESIZE_SIZE)
gt_mask_np = np.array(mask_resized) > 127
```

### Notes and pitfalls

- `--dataset_name` is not used to load data.
- Masks are treated as binary after thresholding; multi-class masks are not supported.
- Ensure image basenames match mask basenames (apart from `.jpg → .png`).


