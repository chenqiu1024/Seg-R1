## seg-rl: Heatmap-based Point Localization (pretraining-ready)

This package provides a clean training/inference pipeline for mapping an image to a single point coordinate using heatmap classification. It is designed to be easily extended and to serve as a pretraining stage before RL fine-tuning.

### Data format

Use JSONL for supervision, one object per line:

```json
{"image": "/abs/path/img_0001.jpg", "x": 123, "y": 456}
```

- Coordinates are in pixel space of the original image. The loader resizes images to H×W and scales coordinates accordingly.

### Quickstart

Train:

```bash
python -m seg_rl.train --train_jsonl /abs/path/train.jsonl \
  --val_jsonl /abs/path/val.jsonl --height 512 --width 512 \
  --batch_size 16 --epochs 20 --loss ce --amp --out_dir /abs/path/outputs
```

Infer:

```bash
python -m seg_rl.infer --ckpt /abs/path/outputs/model_epoch_20.pt \
  --images /abs/path/images_dir --height 512 --width 512 --soft
```

### Modules

- seg_rl.datasets: JSONL dataset, resize with coordinate transforms, simple flips.
- seg_rl.model: ResNet18 backbone + 1-channel heatmap head; argmax/soft-argmax.
- seg_rl.losses: CE over pixels; KL to Gaussian soft targets.
- seg_rl.utils: Checkpoint I/O, PCK metric, simple visualization.

### Notes

- For high-res inputs, consider a higher-resolution backbone (HRNet/UNet) or a coarse-to-fine two-stage model.
- For RL fine-tuning, reuse the same logits as a discrete policy over pixels; add a value head.


