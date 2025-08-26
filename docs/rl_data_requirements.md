### RL input data: format and folder structure

This document describes the exact dataset layout and file format required by the RL stage (invoked via `bash scripts/run_grpo.sh`).

### What the script passes to training

- **images roots (comma-separated)**: `--dataset_image`
- **masks roots (comma-separated)**: `--dataset_gt`

From the default script:

```bash
--dataset_image datasets/COD10K-v3/Train/Image,datasets/CAMO-V.1.0-CVIU2019/Images/Train \
--dataset_gt    datasets/COD10K-v3/Train/GT_Object,datasets/CAMO-V.1.0-CVIU2019/GT/Train
```

Multiple datasets can be provided by separating directories with commas. The loader will build a single training set by concatenating the images from all `--dataset_image` roots, then match each image to a mask by filename in one of the `--dataset_gt` roots.

### Required directory layout

For each dataset included in RL training, provide two leaf directories:

```
<DATASET_ROOT>/
├── <IMAGE_DIR>/        # one of the items in --dataset_image
└── <GT_DIR>/           # one of the items in --dataset_gt
```

You can use any names, but the RL script expects examples like:

```
datasets/COD10K-v3/Train/
├── Image/
└── GT_Object/

datasets/CAMO-V.1.0-CVIU2019/
├── Images/Train/
└── GT/Train/
```

Only the leaf directories matter to the code.

### File naming and 1:1 pairing requirement

- Every image file across all `--dataset_image` directories must have a matching mask file in one of the `--dataset_gt` directories.
- The correspondence is by basename with a fixed extension mapping of `.jpg → .png`.
  - Example: `Image/foo/bar_001.jpg` → `GT_Object/bar_001.png`
  - The loader attempts each `--dataset_gt` root in order; it picks the first existing path. If none exists, it falls back to constructing the path in the first `--dataset_gt` root.
  - If your images are not `.jpg`, masks will not be found because the code replaces `.jpg` with `.png`. Rename images to `.jpg` to comply.
  - Non-recursive scan: each `--dataset_image` directory is scanned with `os.listdir(...)` (non-recursive). Put image files directly inside the specified directory (no nested subfolders are read).

### Accepted formats and preprocessing

- **Images**: Read via Hugging Face Datasets `imagefolder` and converted to RGB. Resized to 768×768. Aspect ratio is not preserved.
- **Masks**: Single-channel grayscale `.png`. Loaded, resized to 768×768, then binarized with threshold `> 127`.
  - Foreground: values > 127
  - Background: values ≤ 127
- The reward code can consume multiple bounding boxes and/or points; this does not affect the on-disk dataset format (still image + binary mask).

### Minimal multi-dataset example

```
datasets/COD10K-v3/Train/
├── Image/
│   ├── A_0001.jpg
│   └── A_0002.jpg
└── GT_Object/
    ├── A_0001.png
    └── A_0002.png

datasets/CAMO-V.1.0-CVIU2019/
├── Images/Train/
│   ├── B_0001.jpg
│   └── B_0002.jpg
└── GT/Train/
    ├── B_0001.png
    └── B_0002.png
```

Run:

```bash
bash scripts/run_grpo.sh \
  --dataset_image datasets/COD10K-v3/Train/Image,datasets/CAMO-V.1.0-CVIU2019/Images/Train \
  --dataset_gt    datasets/COD10K-v3/Train/GT_Object,datasets/CAMO-V.1.0-CVIU2019/GT/Train
```

### How the paths are used in code

Concatenating image folders and deriving mask paths:

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

- The RL code does not use `--dataset_name` for data loading.
- Masks are strictly binary in training after thresholding; multi-class segmentation masks are not supported.
- Provide images as `.jpg` and masks as `.png` with matching basenames to avoid missing-pair issues.


