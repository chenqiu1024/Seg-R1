### Pre-RL input data: format and folder structure

This document describes exactly how the Pre-RL stage (invoked via `bash scripts/run_prerl.sh`) expects its input data to be organized and formatted.

### What the script passes to training

- **images root**: `--dataset_image`
- **masks root**: `--dataset_gt`

From the default script:

```bash
--dataset_image datasets/DIS5K/DIS-TR/im \
--dataset_gt    datasets/DIS5K/DIS-TR/gt
```

Training reads images using Hugging Face Datasets' `imagefolder` builder and then derives each image's mask path from the image filename.

### Required directory layout

Two sibling directories are required:

```
datasets/
└── YOUR_DATASET/
    └── TRAIN_SPLIT/        # any name; not required to be "train"
        ├── im/            # passed as --dataset_image
        └── gt/            # passed as --dataset_gt
```

Only the two leaf directories matter for the code. They can be anywhere; the defaults above are an example.

### File naming and 1:1 pairing requirement

- Every image file in `--dataset_image` must have a corresponding mask file in `--dataset_gt`.
- The correspondence is by basename, with a hardcoded extension mapping of `.jpg → .png`.
  - Example: `im/sample_001.jpg` → `gt/sample_001.png`
  - If your images are not `.jpg`, the current Pre-RL code will fail to locate the masks (because it performs a fixed string replacement of `.jpg` with `.png`). Rename images to `.jpg` (or adapt the paths you pass to match this convention).

### Accepted formats and preprocessing

- **Images**: Read as RGB and resized to 768×768 during training. Aspect ratio is not preserved (stretched resize).
- **Masks**: Must be single-channel grayscale `.png` files. They are loaded as 8-bit, resized to 768×768, then binarized with threshold `> 127`.
  - Foreground: values > 127
  - Background: values ≤ 127

### Minimal example

```
datasets/DIS5K/DIS-TR/
├── im/
│   ├── 0001.jpg
│   ├── 0002.jpg
│   └── 0003.jpg
└── gt/
    ├── 0001.png
    ├── 0002.png
    └── 0003.png
```

Run:

```bash
bash scripts/run_prerl.sh \
  --dataset_image datasets/DIS5K/DIS-TR/im \
  --dataset_gt    datasets/DIS5K/DIS-TR/gt
```

### How the paths are used in code

Image loading and mask path derivation (excerpt):

```330:357:seg-r1/src/open_r1/grpo_prerl.py
dataset = load_dataset(
    "imagefolder",
    data_dir=script_args.dataset_image,
    split=script_args.dataset_train_split,
)

# Add ground truth paths and resize
dataset = dataset.map(
    lambda x: {"gt_path": os.path.join(
        script_args.dataset_gt,
        os.path.basename(x["image"].filename).replace(".jpg", ".png")
    )}
).map(resize_image_and_mask)
```

Mask loading, resizing, and binarization (excerpt):

```41:49:seg-r1/src/open_r1/grpo_prerl.py
mask = PILImage.open(example["gt_path"]).convert("L")
mask_resized = TF.resize(mask, RESIZE_SIZE)
gt_mask_np = (np.array(mask_resized) > 127).astype(np.uint8)
```

### Notes and pitfalls

- `--dataset_train_split` defaults to `train` and is only used to name the split in memory; you do not need a `train/` subfolder on disk.
- `--dataset_name` in the script is currently unused by `grpo_prerl.py`.
- Ensure mask basenames match the image basenames exactly (apart from the `.jpg → .png` extension change).
- Multi-class masks are not supported in Pre-RL; masks are binarized.


