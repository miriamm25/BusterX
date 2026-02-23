#!/usr/bin/env python3
"""
prepare_deepfake_eval.py
========================
Converts Deepfake-Eval-2024 CSV manifest into JSONL files that
train_sft_lora.py and eval_on_test.py expect.

Reads  : Deepfake-Eval-2024/manifests/clips_std_labeled_manifest_with_val.csv
Writes : data/train.jsonl
         data/val.jsonl
         data/test.jsonl

Each output line:
  {"video_path": "/abs/path.mp4", "label": 0, "label_text": "real",
   "video_id": "...", "clip_id": "...", "source": "Deepfake-Eval-2024/train"}

Filters applied:
  1. ffmpeg_rc == 0  (clip was processed successfully)
  2. os.path.exists(video_path)  (file actually on disk; warns and skips if not)

Usage:
  python prepare_deepfake_eval.py [--dataset_root X] [--manifest Y] [--out_dir Z]
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Defaults (server paths)
# ---------------------------------------------------------------------------
DEFAULT_DATASET_ROOT = (
    "/cai3_ds_vm/downloads/deepfake_analysis/Deepfake-Eval-2024"
)
DEFAULT_MANIFEST = (
    DEFAULT_DATASET_ROOT
    + "/manifests/clips_std_labeled_manifest_with_val.csv"
)
DEFAULT_OUT_DIR = str(Path(__file__).parent / "data")

LABEL_MAP = {"real": 0, "fake": 1}


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def build_jsonl(df: pd.DataFrame, dataset_root: str, split_name: str):
    """Yield JSONL dicts for one split, filtering bad rows."""
    rows = df[df["split"] == split_name]
    n_total = len(rows)
    n_missing = 0
    records = []

    for _, row in rows.iterrows():
        video_path = os.path.join(dataset_root, row["dst_rel_path"])
        if not os.path.exists(video_path):
            n_missing += 1
            continue

        label_text = str(row["label"]).lower()
        label_int = LABEL_MAP.get(label_text)
        if label_int is None:
            # Unexpected label value – skip with warning
            print(f"  [WARN] unknown label '{row['label']}' in clip {row['clip_id']} – skipped")
            n_missing += 1
            continue

        records.append({
            "video_path": video_path,
            "label": label_int,
            "label_text": label_text,
            "video_id": str(row["video_id"]),
            "clip_id": str(row["clip_id"]),
            "source": f"Deepfake-Eval-2024/{split_name}",
        })

    return records, n_total, n_missing


def write_jsonl(records: list, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def print_stats(split_name: str, records: list, n_total: int, n_missing: int):
    n_real = sum(1 for r in records if r["label"] == 0)
    n_fake = sum(1 for r in records if r["label"] == 1)
    print(
        f"  {split_name:<6}  total={n_total:<6}  written={len(records):<6}"
        f"  real={n_real:<6}  fake={n_fake:<6}  missing/skipped={n_missing}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build SFT JSONL manifests from Deepfake-Eval-2024 CSV"
    )
    parser.add_argument(
        "--dataset_root", default=DEFAULT_DATASET_ROOT,
        help="Absolute path to Deepfake-Eval-2024 dataset root on server",
    )
    parser.add_argument(
        "--manifest", default=DEFAULT_MANIFEST,
        help="Path to clips_std_labeled_manifest_with_val.csv",
    )
    parser.add_argument(
        "--out_dir", default=DEFAULT_OUT_DIR,
        help="Output directory for train/val/test JSONL files",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Deepfake-Eval-2024  →  SFT JSONL")
    print("=" * 60)
    print(f"  dataset_root : {args.dataset_root}")
    print(f"  manifest     : {args.manifest}")
    print(f"  out_dir      : {args.out_dir}")
    print()

    # Load manifest
    df = pd.read_csv(args.manifest)
    print(f"Manifest loaded: {len(df)} rows, splits = {sorted(df['split'].unique())}")

    # Filter ffmpeg_rc == 0
    n_before = len(df)
    df = df[df["ffmpeg_rc"] == 0].copy()
    n_filtered = n_before - len(df)
    if n_filtered:
        print(f"  Dropped {n_filtered} rows with ffmpeg_rc != 0")

    # Build and write each split
    print("\nBuilding splits:")
    for split_name, out_file in [
        ("train", os.path.join(args.out_dir, "train.jsonl")),
        ("val",   os.path.join(args.out_dir, "val.jsonl")),
        ("test",  os.path.join(args.out_dir, "test.jsonl")),
    ]:
        records, n_total, n_missing = build_jsonl(df, args.dataset_root, split_name)
        write_jsonl(records, out_file)
        print_stats(split_name, records, n_total, n_missing)

    print()
    print("Done. JSONL files written to:", args.out_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
