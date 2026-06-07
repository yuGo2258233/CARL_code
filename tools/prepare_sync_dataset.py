#!/usr/bin/env python3
"""Prepare a custom mp4 folder into CARL Pouring-style dataset files.

This script creates:
  - <output_dir>/videos/*.mp4
  - <output_dir>/train.pkl
  - <output_dir>/val.pkl

Each entry follows the schema used by datasets/pouring.py.
"""

import argparse
import os
import pickle
import shutil
from typing import Dict, List, Tuple

import cv2
import torch


def list_mp4_files(input_dir: str) -> List[str]:
    files = []
    for name in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, name)
        if os.path.isfile(path) and name.lower().endswith(".mp4"):
            files.append(path)
    return files


def build_item(idx: int, rel_video_path: str, video_name: str, seq_len: int) -> Dict:
    # Frame labels are only used as a validity mask in this workflow.
    frame_label = torch.zeros(seq_len, dtype=torch.long)
    return {
        "id": idx,
        "video_file": rel_video_path,
        "frame_label": frame_label,
        "seq_len": seq_len,
        "name": video_name,
    }


def split_train_val(items: List[Dict], val_ratio: float) -> Tuple[List[Dict], List[Dict]]:
    if len(items) < 2:
        raise ValueError("Need at least 2 videos to compute synchronization.")
    n_val = max(1, int(round(len(items) * val_ratio)))
    n_val = min(n_val, len(items) - 1)
    val_items = items[:n_val]
    train_items = items[n_val:]
    return train_items, val_items


def get_num_frames(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if num_frames <= 0:
        raise ValueError(f"Failed to read frame count: {video_path}")
    return num_frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare custom dataset for CARL sync inference.")
    parser.add_argument("--input_dir", required=True, help="Directory containing source .mp4 files.")
    parser.add_argument("--output_dir", required=True, help="Output dataset directory.")
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.5,
        help="Validation split ratio. Default: 0.5",
    )
    parser.add_argument(
        "--copy_videos",
        action="store_true",
        help="Copy videos into output_dir/videos. If false, uses relative paths if possible.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    videos_dir = os.path.join(args.output_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)

    mp4_files = list_mp4_files(args.input_dir)
    if not mp4_files:
        raise ValueError(f"No .mp4 files found in {args.input_dir}")

    items = []
    for idx, src_path in enumerate(mp4_files):
        base_name = os.path.basename(src_path)
        stem = os.path.splitext(base_name)[0]

        if args.copy_videos:
            dst_path = os.path.join(videos_dir, base_name)
            if os.path.abspath(src_path) != os.path.abspath(dst_path):
                shutil.copy2(src_path, dst_path)
            rel_video_path = os.path.join("videos", base_name)
            video_path_for_read = dst_path
        else:
            # Keep paths relative to output_dir whenever possible.
            rel_video_path = os.path.relpath(src_path, args.output_dir)
            video_path_for_read = src_path

        seq_len = get_num_frames(video_path_for_read)
        if seq_len <= 1:
            raise ValueError(f"Video too short for sync: {src_path} (frames={seq_len})")

        item = build_item(idx, rel_video_path, stem, seq_len)
        items.append(item)
        print(f"[OK] {base_name}: {seq_len} frames")

    train_items, val_items = split_train_val(items, args.val_ratio)

    train_pkl = os.path.join(args.output_dir, "train.pkl")
    val_pkl = os.path.join(args.output_dir, "val.pkl")
    with open(train_pkl, "wb") as f:
        pickle.dump(train_items, f)
    with open(val_pkl, "wb") as f:
        pickle.dump(val_items, f)

    print("\nDone.")
    print(f"train.pkl: {train_pkl} ({len(train_items)} videos)")
    print(f"val.pkl:   {val_pkl} ({len(val_items)} videos)")
    print(f"dataset root: {args.output_dir}")


if __name__ == "__main__":
    main()
