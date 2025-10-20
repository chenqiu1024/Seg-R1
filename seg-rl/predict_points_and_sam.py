#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _abspath(path: str) -> str:
    return os.path.abspath(path)


@dataclass
class Paths:
    task_name: str
    base_out: str
    jsonl: str
    sam_masks_dir: str
    dbg_points_dir: str


def build_paths(task_name: str) -> Paths:
    base_out = os.path.join("outputs", "braintumour")
    jsonl = os.path.join(base_out, f"{task_name}.jsonl")
    sam_masks_dir = os.path.join(base_out, f"{task_name}_sam_masks")
    dbg_points_dir = os.path.join(base_out, f"{task_name}_dbg_pred_points")
    return Paths(task_name, base_out, jsonl, sam_masks_dir, dbg_points_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Python driver for predicting points, running SAM, evaluating, and plotting metrics vs prompt count")
    p.add_argument("task_name", type=str, help="Name of task, used to name outputs")
    p.add_argument("n_points", type=int, help="Total number of prompts per image (>=1)")
    p.add_argument("model_path", type=str, help="Path to point predictor model checkpoint (.pt)")
    p.add_argument("device", type=str, help="Device for SAM (e.g., cuda, cuda:0, cpu)")

    # Optional overrides
    p.add_argument("--images_dir", type=str, default="datasets/seg_r1_md/Task01_BrainTumour/canonical/images")
    p.add_argument("--masks_dir", type=str, default="datasets/seg_r1_md/Task01_BrainTumour/canonical/masks")
    p.add_argument("--sam_checkpoint", type=str, default="third_party/sam2/checkpoints/sam2.1_hiera_large.pt")
    p.add_argument("--resize", type=int, nargs=2, default=[512, 512], metavar=("H", "W"))

    # Existing-file handling
    g = p.add_mutually_exclusive_group()
    g.add_argument("--delete-existing", action="store_true", help="If set, delete existing JSONL and output dirs without prompting")
    g.add_argument("--keep-existing", action="store_true", help="If set and JSONL exists, skip initial first-prompt generation")
    p.add_argument("--non-interactive", action="store_true", help="Disable interactive prompts; requires one of --delete-existing/--keep-existing if outputs exist")

    # Plot config
    p.add_argument("--save_csv", action="store_true", help="Also save per-round metrics to CSV next to plot")

    return p.parse_args()


def _confirm_and_prepare(paths: Paths, delete_existing: bool, keep_existing: bool, non_interactive: bool) -> Tuple[bool, List[str]]:
    """Check existing outputs. Return (skip_first, deleted_list)."""
    existing: List[str] = []
    if os.path.isfile(paths.jsonl):
        existing.append(paths.jsonl)
    if os.path.isdir(paths.sam_masks_dir):
        existing.append(paths.sam_masks_dir)
    if os.path.isdir(paths.dbg_points_dir):
        existing.append(paths.dbg_points_dir)

    skip_first = False
    deleted: List[str] = []

    if not existing:
        return False, deleted

    print("\nWarning: The following files/folders already exist:")
    for p in existing:
        print(f"  - {p}")

    if delete_existing:
        print("Deleting existing files/folders (per --delete-existing)...")
        for p in existing:
            try:
                if os.path.isfile(p):
                    os.remove(p)
                    print(f"  Deleted file: {p}")
                elif os.path.isdir(p):
                    shutil.rmtree(p)
                    print(f"  Deleted directory: {p}")
                deleted.append(p)
            except Exception as e:
                print(f"  Failed to delete {p}: {e}")
        print("Cleanup completed. Proceeding...")
        return False, deleted

    if keep_existing or non_interactive:
        print("Keeping existing files/folders. Proceeding without deletion...")
        if os.path.isfile(paths.jsonl):
            skip_first = True
            print(f"Detected existing JSONL ({paths.jsonl}). Will skip initial first-prompt generation.")
        return skip_first, deleted

    # Interactive prompt
    try:
        ans = input("Do you want to delete these existing files/folders and continue? (y/N): ").strip()
    except EOFError:
        ans = ""
    if ans.lower().startswith("y"):
        print("Deleting existing files/folders...")
        for p in existing:
            try:
                if os.path.isfile(p):
                    os.remove(p)
                    print(f"  Deleted file: {p}")
                elif os.path.isdir(p):
                    shutil.rmtree(p)
                    print(f"  Deleted directory: {p}")
                deleted.append(p)
            except Exception as e:
                print(f"  Failed to delete {p}: {e}")
        print("Cleanup completed. Proceeding...")
        return False, deleted
    else:
        print("Keeping existing files/folders. Proceeding without deletion...")
        if os.path.isfile(paths.jsonl):
            skip_first = True
            print(f"Detected existing JSONL ({paths.jsonl}). Will skip initial first-prompt generation.")
        return skip_first, deleted


def _run(cmd: List[str], capture: bool = False) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd))
    if capture:
        return subprocess.run(cmd, check=True, text=True, capture_output=True)
    return subprocess.run(cmd, check=True)


def _parse_dataset_averages(stdout_text: str) -> Optional[Dict[str, float]]:
    # Find "Dataset averages:" block and parse lines like "  DICE: 0.4790"
    lines = stdout_text.splitlines()
    try:
        start = lines.index("Dataset averages:")
    except ValueError:
        # Some runs might not print averages if no samples processed
        return None
    metrics: Dict[str, float] = {}
    for i in range(start + 1, len(lines)):
        line = lines[i].strip()
        if not line or line.startswith("Done."):
            break
        m = re.match(r"^([A-Z_]+):\s*([0-9]*\.?[0-9]+)$", line)
        if m:
            key = m.group(1)
            val = float(m.group(2))
            metrics[key] = val
        else:
            # Stop at first non-metric line in the block
            if ":" in line:
                # tolerate lines like warnings
                continue
            break
    return metrics or None


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> int:
    args = parse_args()
    if args.n_points < 1:
        print("Error: n_points must be >= 1")
        return 2

    py = sys.executable
    paths = build_paths(args.task_name)
    _ensure_dir(paths.base_out)

    # Handle existing outputs and decide whether to skip first block
    skip_first, _ = _confirm_and_prepare(paths, args.delete_existing, args.keep_existing, args.non_interactive)

    metrics_over_rounds: Dict[int, Dict[str, float]] = {}

    # Initial first-prompt generation (round 1)
    if not skip_first:
        print("Generate the first point prompt for each image:")
        _run([
            py, "-m", "seg-rl.heatmap.predict_next_point_from_model",
            "--model_path", args.model_path,
            "--images_dir", args.images_dir,
            "--masks_dir", args.masks_dir,
            "--output_json", paths.jsonl,
        ])

        _ensure_dir(paths.sam_masks_dir)
        _run([
            py, "seg-rl/sam2_segment_from_points.py",
            "--input_jsonl", paths.jsonl,
            "--json_output", paths.jsonl,
            "--output_dir", paths.sam_masks_dir,
            "--sam_checkpoint", args.sam_checkpoint,
            "--device", args.device,
            "--resize", str(args.resize[0]), str(args.resize[1]),
            "--skip_existing",
        ])

        # Evaluate round 1 and capture dataset averages
        proc = _run([
            py, "seg-rl/evaluation/eval_sam_masks.py",
            "--input_json", paths.jsonl,
            "--num_prompts", "1",
        ], capture=True)
        avg1 = _parse_dataset_averages(proc.stdout)
        if avg1:
            metrics_over_rounds[1] = avg1

    # Loop from 2..n_points
    print(f"Starting loop from 1 to {args.n_points - 1}:")
    for i in range(1, args.n_points):
        print(f"Generate the {i+1}th point prompt for each image")
        _run([
            py, "-m", "seg-rl.heatmap.predict_next_point_from_model",
            "--model_path", args.model_path,
            "--appendto_json", paths.jsonl,
            "--sam_dir", paths.sam_masks_dir,
        ])

        _run([
            py, "seg-rl/sam2_segment_from_points.py",
            "--input_jsonl", paths.jsonl,
            "--json_output", paths.jsonl,
            "--output_dir", paths.sam_masks_dir,
            "--sam_checkpoint", args.sam_checkpoint,
            "--device", args.device,
            "--resize", str(args.resize[0]), str(args.resize[1]),
            "--skip_existing",
        ])

        # Evaluate round i+1 and capture dataset averages
        proc = _run([
            py, "seg-rl/evaluation/eval_sam_masks.py",
            "--input_json", paths.jsonl,
            "--num_prompts", str(i + 1),
        ], capture=True)
        avg = _parse_dataset_averages(proc.stdout)
        if avg:
            metrics_over_rounds[i + 1] = avg

    # Post evaluation similar to shell script
    print("Evaluate the performance of the model")
    _ensure_dir(paths.dbg_points_dir)
    _run([
        py, "seg-rl/annotator/gen_point_jsonl_from_masks.py",
        "--debug_json", paths.jsonl,
        "--debug_output_dir", paths.dbg_points_dir,
    ])

    _run([
        py, "seg-rl/sam2_automatic_evaluation.py",
        "--input_jsonl", paths.jsonl,
        "--sam_masks_dir", paths.sam_masks_dir,
        "--sam_checkpoint", args.sam_checkpoint,
        "--device", args.device,
    ])

    # Save metrics and plot curves
    if metrics_over_rounds:
        out_dir = paths.base_out
        metrics_path = os.path.join(out_dir, f"{paths.task_name}_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in metrics_over_rounds.items()}, f, indent=2, ensure_ascii=False)
        print(f"Saved per-round dataset-average metrics to {metrics_path}")

        if args.save_csv:
            csv_path = os.path.join(out_dir, f"{paths.task_name}_metrics.csv")
            # Collect all keys
            keys: List[str] = sorted(next(iter(metrics_over_rounds.values())).keys())
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write(",".join(["prompts"] + keys) + "\n")
                for k in sorted(metrics_over_rounds.keys()):
                    row = [str(k)] + [f"{metrics_over_rounds[k].get(key, float('nan')):.6f}" for key in keys]
                    f.write(",".join(row) + "\n")
            print(f"Saved CSV to {csv_path}")

        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"Plotting skipped (matplotlib not available): {e}")
        else:
            keys: List[str] = sorted(next(iter(metrics_over_rounds.values())).keys())
            xs = sorted(metrics_over_rounds.keys())
            plt.figure(figsize=(8, 5))
            for key in keys:
                ys = [metrics_over_rounds.get(x, {}).get(key, float("nan")) for x in xs]
                plt.plot(xs, ys, marker="o", label=key)
            plt.xlabel("Number of prompts")
            plt.ylabel("Dataset-average metric value")
            plt.title(paths.task_name)
            plt.grid(True, linestyle=":", alpha=0.4)
            plt.legend()
            plot_path = os.path.join(paths.base_out, f"{paths.task_name}_metrics.png")
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"Saved plot to {plot_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


