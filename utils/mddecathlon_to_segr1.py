#!/usr/bin/env python3

"""
Medical Decathlon -> Seg-R1 dataset converter

This utility converts 3D NIfTI volumes (.nii or .nii.gz) into 2D slice image/mask pairs
that comply with Seg-R1 training data requirements for Pre-RL, RL, and SOD fine-tune.

Key behaviors
- Reads image volumes from a Medical Decathlon-like structure (e.g., imagesTr/labelsTr).
- Converts each selected slice to an 8-bit .jpg image and a binary .png mask.
- Names slices with a stable basename that includes the slice index (e.g., case123_z0007.jpg).
- Writes ONE canonical set of real files per version, then creates symlinked "views" that
  match each training stage’s expected leaf directory naming.

Storage and linking rules
- The original source files are never modified.
- Canonical outputs are real image files. Views for different training stages are symlinks
  pointing back to the canonical files to minimize storage.
- Different versions (e.g., size variants) are stored in separate canonical folders.

Example
  python utils/mddecathlon_to_segr1.py \
    --input_root datasets/medical_decathlon/Task09_Spleen \
    --output_root datasets/seg_r1_md \
    --task_name Task09_Spleen \
    --slice_axis 2 --keep_empty False --size_variants  \
    canonical,512 --intensity percentiles:1,99

  python utils/mddecathlon_to_segr1.py \
    --input_root /root/autodl-tmp/works/Seg-R0/datasets/medical_decathlon/Task04_Hippocampus \
    --output_root /root/autodl-tmp/works/Seg-R0/datasets/seg_r1_md \
    --task_name Task04_Hippocampus \
    --images_subdir imagesTr \
    --labels_subdir labelsTr \
    --slice_axis 2 \
    --keep_empty False \
    --size_variants canonical,256,512 \
    --intensity percentiles:2,98 \
    --case_id_regex "hippocampus_(\d+)"

After running, you can point scripts to the created views, e.g.:
- Pre-RL:   --dataset_image datasets/seg_r1_md/Task09_Spleen/views/prerl/im \
            --dataset_gt    datasets/seg_r1_md/Task09_Spleen/views/prerl/gt
- RL:       --dataset_image datasets/seg_r1_md/Task09_Spleen/views/rl/Image \
            --dataset_gt    datasets/seg_r1_md/Task09_Spleen/views/rl/GT_Object
- SOD FT:   --dataset_image datasets/seg_r1_md/Task09_Spleen/views/sod/DUTS-TR-Image \
            --dataset_gt    datasets/seg_r1_md/Task09_Spleen/views/sod/DUTS-TR-Mask

Dependencies: nibabel, numpy, pillow
Install (if needed): pip install nibabel numpy pillow
"""

import argparse
import os
import re
import sys
import math
import shutil
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import nibabel as nib  # Preferred for NIfTI
except Exception as e:  # pragma: no cover
    nib = None


# -----------------------------
# Utility helpers
# -----------------------------

def ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def make_relative_symlink(target_path: str, link_path: str) -> None:
    """Create a relative symlink at link_path pointing to target_path.

    Overwrites existing symlink/file if it exists and points elsewhere.
    """
    if os.path.islink(link_path) or os.path.exists(link_path):
        # If an identical symlink already exists, keep it. Otherwise, replace.
        try:
            if os.path.islink(link_path):
                existing = os.readlink(link_path)
                # If the link already points to the correct target (relative or absolute), keep it
                if os.path.normpath(os.path.join(os.path.dirname(link_path), existing)) == os.path.normpath(target_path):
                    return
            # Remove wrong target or existing file
            os.remove(link_path)
        except OSError:
            # As a fallback, attempt to unlink via shutil
            try:
                if os.path.isdir(link_path) and not os.path.islink(link_path):
                    shutil.rmtree(link_path)
                else:
                    os.remove(link_path)
            except Exception:
                pass

    rel_target = os.path.relpath(target_path, start=os.path.dirname(link_path))
    os.symlink(rel_target, link_path)


def binarize_mask(mask_slice: np.ndarray) -> np.ndarray:
    """Convert any non-zero label to 255, zero stays 0, dtype uint8."""
    binary = (mask_slice.astype(np.int64) != 0).astype(np.uint8) * 255
    return binary


def scale_to_uint8(
    slice_data: np.ndarray,
    method: str = "percentiles",
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> np.ndarray:
    """Scale a single 2D slice to uint8.

    - method="percentiles": robust min-max using given percentiles (default 1–99).
    - method="minmax": direct min-max over slice.
    - method="zscore": z-score then 3-sigma clip, mapped to 0–255.
    """
    data = slice_data.astype(np.float32)

    if method == "percentiles":
        lo = np.percentile(data, p_low)
        hi = np.percentile(data, p_high)
        if hi <= lo:
            lo, hi = float(data.min()), float(data.max())
        data = np.clip((data - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        data = (data * 255.0).round().astype(np.uint8)
        return data

    if method == "minmax":
        lo, hi = float(data.min()), float(data.max())
        if hi <= lo:
            return np.zeros_like(data, dtype=np.uint8)
        data = (data - lo) / (hi - lo)
        data = (data * 255.0).round().astype(np.uint8)
        return data

    if method == "zscore":
        mean = float(data.mean())
        std = float(data.std() + 1e-6)
        data = (data - mean) / std
        data = np.clip((data + 3.0) / 6.0, 0.0, 1.0)  # map [-3,3] -> [0,1]
        data = (data * 255.0).round().astype(np.uint8)
        return data

    raise ValueError(f"Unknown intensity scaling method: {method}")


def save_jpg(pixels: np.ndarray, out_path: str) -> None:
    Image.fromarray(pixels).convert("RGB").save(out_path, format="JPEG", quality=95)


def save_png_mask(mask: np.ndarray, out_path: str) -> None:
    Image.fromarray(mask).save(out_path, format="PNG", optimize=True)


# -----------------------------
# Conversion configuration
# -----------------------------

@dataclass
class ConversionConfig:
    input_root: str
    output_root: str
    task_name: str
    images_subdir: str = "imagesTr"  # typical decathlon train images folder
    labels_subdir: str = "labelsTr"  # typical decathlon train labels folder
    slice_axis: int = 2               # 0=sagittal, 1=coronal, 2=axial
    keep_empty_slices: bool = False   # if False, drop slices with empty mask
    intensity_mode: str = "percentiles"  # one of: percentiles, minmax, zscore
    intensity_percentiles: Tuple[float, float] = (1.0, 99.0)
    size_variants: Tuple[Optional[int], ...] = (None,)  # e.g., (None, 512) -> canonical, 512px square
    jpg_quality: int = 95


# -----------------------------
# Core conversion
# -----------------------------

def remove_dotfiles_in_dir(target_dir: Optional[str], recursive: bool = False) -> int:
    """Remove files whose basename starts with '.' under target_dir.

    Returns the number of files removed. If directory is missing, returns 0.
    """
    if not target_dir or not os.path.isdir(target_dir):
        return 0
    removed = 0
    if recursive:
        for dirpath, dirnames, filenames in os.walk(target_dir):
            for fname in filenames:
                if fname.startswith('.'):
                    fpath = os.path.join(dirpath, fname)
                    try:
                        os.remove(fpath)
                        removed += 1
                    except Exception:
                        # Ignore files that cannot be removed
                        pass
    else:
        for fname in os.listdir(target_dir):
            if fname.startswith('.'):
                fpath = os.path.join(target_dir, fname)
                if os.path.isfile(fpath) or os.path.islink(fpath):
                    try:
                        os.remove(fpath)
                        removed += 1
                    except Exception:
                        pass
    return removed

def _is_valid_nii_name(filename: str) -> bool:
    """Return True if filename is a valid NIfTI file name we should process.

    Skips dotfiles and AppleDouble metadata files like '._name.nii.gz'.
    """
    if filename.startswith('.') or filename.startswith('._'):
        return False
    return filename.endswith('.nii') or filename.endswith('.nii.gz')


def find_decathlon_pairs(images_dir: str, labels_dir: Optional[str]) -> List[Tuple[str, Optional[str]]]:
    """Find (image_path, label_path) pairs by matching basenames (without extensions).

    Supports files ending with .nii or .nii.gz.
    """
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    label_lookup = {}
    if labels_dir and os.path.isdir(labels_dir):
        for f in os.listdir(labels_dir):
            if _is_valid_nii_name(f):
                stem = re.sub(r"\.nii(\.gz)?$", "", f)
                label_lookup[stem] = os.path.join(labels_dir, f)

    pairs: List[Tuple[str, Optional[str]]] = []
    for f in os.listdir(images_dir):
        if _is_valid_nii_name(f):
            stem = re.sub(r"\.nii(\.gz)?$", "", f)
            img_path = os.path.join(images_dir, f)
            lbl_path = label_lookup.get(stem)
            pairs.append((img_path, lbl_path))
    return sorted(pairs)


def load_nii(path: str) -> np.ndarray:
    if nib is None:
        raise ImportError("nibabel is required to read NIfTI files. Please install it: pip install nibabel")
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    return data


def iter_slices(volume: np.ndarray, axis: int) -> Iterable[np.ndarray]:
    num_slices = volume.shape[axis]
    for idx in range(num_slices):
        yield np.take(volume, indices=idx, axis=axis)


def resize_square_uint8(img_u8: np.ndarray, side: int) -> np.ndarray:
    """Resize a 2D uint8 array to side x side using bilinear (images) or nearest (masks)."""
    pil = Image.fromarray(img_u8)
    pil = pil.resize((side, side), resample=Image.BILINEAR)
    return np.array(pil, dtype=np.uint8)


def resize_square_mask(mask_u8: np.ndarray, side: int) -> np.ndarray:
    pil = Image.fromarray(mask_u8)
    pil = pil.resize((side, side), resample=Image.NEAREST)
    return np.array(pil, dtype=np.uint8)


def slice_and_write(
    case_id: str,
    image_vol: np.ndarray,
    label_vol: Optional[np.ndarray],
    cfg: ConversionConfig,
    canonical_images_dir: str,
    canonical_labels_dir: Optional[str],
    version_name: str,
    resize_side: Optional[int],
) -> Tuple[int, int]:
    """Write 2D slices for one case into canonical folder for a given version.

    Returns (num_written_images, num_written_masks)
    """
    written_images = 0
    written_masks = 0
    p_low, p_high = cfg.intensity_percentiles

    num_slices = image_vol.shape[cfg.slice_axis]
    for z, img_slice in enumerate(iter_slices(image_vol, cfg.slice_axis)):
        mask_slice = None
        if label_vol is not None:
            mask_slice = next(iter_slices(label_vol, cfg.slice_axis)) if False else np.take(label_vol, z, axis=cfg.slice_axis)
            mask_bin = binarize_mask(mask_slice)
            if not cfg.keep_empty_slices and mask_bin.sum() == 0:
                continue
        # scale image
        img_u8 = scale_to_uint8(img_slice, method=cfg.intensity_mode, p_low=p_low, p_high=p_high)
        # optional resize
        if resize_side is not None:
            img_u8 = resize_square_uint8(img_u8, resize_side)
            if mask_slice is not None:
                mask_bin = resize_square_mask(mask_bin, resize_side)

        base = f"{case_id}_z{z:04d}"
        img_out = os.path.join(canonical_images_dir, f"{base}.jpg")
        save_jpg(img_u8, img_out)
        written_images += 1

        if mask_slice is not None and canonical_labels_dir is not None:
            msk_out = os.path.join(canonical_labels_dir, f"{base}.png")
            save_png_mask(mask_bin, msk_out)
            written_masks += 1

    return written_images, written_masks


def create_views_for_version(
    version_root: str,
    create_mask_views: bool,
) -> None:
    """Create symlinked leaf directories for Pre-RL, RL, and SOD fine-tune views.

    Views created under: <version_root>/views/{prerl,rl,sod}/...
    """
    images_dir = os.path.join(version_root, "canonical", "images")
    masks_dir = os.path.join(version_root, "canonical", "masks") if create_mask_views else None

    # Pre-RL view
    prerl_im = os.path.join(version_root, "views", "prerl", "im")
    prerl_gt = os.path.join(version_root, "views", "prerl", "gt")
    ensure_dir(prerl_im)
    if create_mask_views:
        ensure_dir(prerl_gt)

    # RL view
    rl_im = os.path.join(version_root, "views", "rl", "Image")
    rl_gt = os.path.join(version_root, "views", "rl", "GT_Object")
    ensure_dir(rl_im)
    if create_mask_views:
        ensure_dir(rl_gt)

    # SOD fine-tune view (DUTS-like leaf names)
    sod_im = os.path.join(version_root, "views", "sod", "DUTS-TR-Image")
    sod_gt = os.path.join(version_root, "views", "sod", "DUTS-TR-Mask")
    ensure_dir(sod_im)
    if create_mask_views:
        ensure_dir(sod_gt)

    # For each image, link to all three views. For masks, same.
    for fname in os.listdir(images_dir):
        if not fname.lower().endswith(".jpg"):
            continue
        src = os.path.join(images_dir, fname)
        make_relative_symlink(src, os.path.join(prerl_im, fname))
        make_relative_symlink(src, os.path.join(rl_im, fname))
        make_relative_symlink(src, os.path.join(sod_im, fname))

    if create_mask_views and masks_dir and os.path.isdir(masks_dir):
        for fname in os.listdir(masks_dir):
            if not fname.lower().endswith(".png"):
                continue
            src = os.path.join(masks_dir, fname)
            make_relative_symlink(src, os.path.join(prerl_gt, fname))
            make_relative_symlink(src, os.path.join(rl_gt, fname))
            make_relative_symlink(src, os.path.join(sod_gt, fname))


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert Medical Decathlon NIfTI volumes into Seg-R1-ready 2D slices and symlinked views.")
    p.add_argument("--input_root", required=True, help="Path to a Task folder (e.g., datasets/medical_decathlon/Task09_Spleen)")
    p.add_argument("--output_root", required=True, help="Base output directory for converted datasets")
    p.add_argument("--task_name", required=False, help="Task name to use under output_root (default: basename of input_root)")
    p.add_argument("--images_subdir", default="imagesTr", help="Subdir for images within input_root (default: imagesTr)")
    p.add_argument("--labels_subdir", default="labelsTr", help="Subdir for labels within input_root (default: labelsTr). If missing, only images are produced.")
    p.add_argument("--slice_axis", type=int, default=2, choices=[0,1,2], help="Axis to slice along: 0=sagittal, 1=coronal, 2=axial")
    p.add_argument("--keep_empty", type=lambda s: s.lower() in {"1","true","yes","y"}, default=False, help="Keep slices with empty masks (default: False)")
    p.add_argument("--intensity", default="percentiles:1,99", help="Scaling mode. Options: 'percentiles:lo,hi' | 'minmax' | 'zscore'")
    p.add_argument("--size_variants", default="canonical,512", help="Comma list of variants. Use 'canonical' for original size, or an integer (e.g., 512) for square resize")
    p.add_argument("--case_id_regex", default=None, help="Optional regex to extract a simpler case id from filename stem")
    return p.parse_args()


def parse_intensity(spec: str) -> Tuple[str, Tuple[float, float]]:
    if spec.startswith("percentiles:"):
        try:
            rest = spec.split(":", 1)[1]
            lo_s, hi_s = rest.split(",")
            return "percentiles", (float(lo_s), float(hi_s))
        except Exception as e:
            raise ValueError("percentiles spec must be percentiles:lo,hi")
    if spec == "minmax":
        return "minmax", (0.0, 100.0)
    if spec == "zscore":
        return "zscore", (0.0, 100.0)
    raise ValueError(f"Invalid intensity spec: {spec}")


def parse_variants(spec: str) -> List[Tuple[str, Optional[int]]]:
    variants: List[Tuple[str, Optional[int]]] = []
    for tok in [t.strip() for t in spec.split(",") if t.strip()]:
        if tok.lower() == "canonical":
            variants.append(("canonical", None))
        else:
            try:
                side = int(tok)
                variants.append((f"size{side}", side))
            except ValueError:
                raise ValueError(f"Invalid size variant token: {tok}")
    return variants


def stem_from_filename(fname: str) -> str:
    return re.sub(r"\.nii(\.gz)?$", "", os.path.basename(fname))


def main() -> None:
    args = parse_args()
    intensity_mode, (p_lo, p_hi) = parse_intensity(args.intensity)
    variants = parse_variants(args.size_variants)

    task_name = args.task_name or os.path.basename(os.path.normpath(args.input_root))
    images_dir = os.path.join(args.input_root, args.images_subdir)
    labels_dir = os.path.join(args.input_root, args.labels_subdir) if args.labels_subdir else None

    # Pre-clean: remove dotfiles in input directories to avoid invalid entries
    removed_images = remove_dotfiles_in_dir(images_dir)
    removed_labels = remove_dotfiles_in_dir(labels_dir) if labels_dir else 0
    if removed_images or removed_labels:
        print(f"Removed dotfiles -> images: {removed_images}, labels: {removed_labels}")

    pairs = find_decathlon_pairs(images_dir, labels_dir)
    if len(pairs) == 0:
        print(f"No NIfTI files found under {images_dir}")
        sys.exit(1)

    cfg = ConversionConfig(
        input_root=args.input_root,
        output_root=args.output_root,
        task_name=task_name,
        images_subdir=args.images_subdir,
        labels_subdir=args.labels_subdir,
        slice_axis=args.slice_axis,
        keep_empty_slices=bool(args.keep_empty),
        intensity_mode=intensity_mode,
        intensity_percentiles=(p_lo, p_hi),
        size_variants=tuple(v[1] for v in variants),
    )

    # Convert per variant
    for (version_name, resize_side) in variants:
        version_root = os.path.join(cfg.output_root, cfg.task_name, version_name)
        canonical_images_dir = os.path.join(version_root, "canonical", "images")
        canonical_labels_dir = os.path.join(version_root, "canonical", "masks") if os.path.isdir(labels_dir or "") else None
        ensure_dir(canonical_images_dir)
        if canonical_labels_dir:
            ensure_dir(canonical_labels_dir)

        total_imgs = 0
        total_msks = 0

        for img_path, lbl_path in pairs:
            case_stem = stem_from_filename(img_path)
            if args.case_id_regex:
                m = re.search(args.case_id_regex, case_stem)
                case_id = m.group(1) if m else case_stem
            else:
                case_id = case_stem

            # Load volumes (skip unreadable files gracefully)
            try:
                img_vol = load_nii(img_path)
            except Exception as e:
                print(f"Skipping unreadable image: {img_path} ({e})")
                continue
            try:
                lbl_vol = load_nii(lbl_path) if (lbl_path and os.path.exists(lbl_path)) else None
            except Exception as e:
                print(f"Skipping label for {img_path}: {lbl_path} ({e})")
                lbl_vol = None

            # Write slices
            n_i, n_m = slice_and_write(
                case_id=case_id,
                image_vol=img_vol,
                label_vol=lbl_vol,
                cfg=cfg,
                canonical_images_dir=canonical_images_dir,
                canonical_labels_dir=canonical_labels_dir,
                version_name=version_name,
                resize_side=resize_side,
            )
            total_imgs += n_i
            total_msks += n_m

        # Create symlinked views
        create_views_for_version(version_root=version_root, create_mask_views=canonical_labels_dir is not None)

        print(f"[Version {version_name}] Wrote {total_imgs} images and {total_msks} masks under {version_root}")


if __name__ == "__main__":
    main()


