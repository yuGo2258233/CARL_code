#!/usr/bin/env python3
"""Run CARL synchronization inference without visualization outputs.

Outputs:
  - pairwise frame mapping (nearest-neighbor path)
  - pairwise Kendall's Tau
  - average Kendall's Tau
"""

import argparse
import os
import pickle
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.stats import kendalltau

from models import build_model
from utils.parser import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync-only inference with CARL.")
    parser.add_argument("--workdir", type=str, required=True, help="Dataset root parent.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset folder name under workdir.")
    parser.add_argument("--cfg_file", type=str, required=True, help="Path to CARL config YAML.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to CARL checkpoint .pth file.")
    parser.add_argument("--output_json", type=str, default=None, help="Optional json output path.")
    parser.add_argument("--stride", type=int, default=5, help="Temporal stride for tau computation.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    return parser.parse_args()


def load_cfg(args: argparse.Namespace):
    # Reuse existing config loader by providing expected fields.
    class Dummy:
        pass

    dummy = Dummy()
    dummy.cfg_file = args.cfg_file
    dummy.opts = None
    dummy.logdir = None
    cfg = load_config(dummy)
    cfg.PATH_TO_DATASET = os.path.join(args.workdir, args.dataset_name)
    return cfg


def load_val_items(dataset_root: str) -> List[Dict]:
    val_path = os.path.join(dataset_root, "val.pkl")
    with open(val_path, "rb") as f:
        items = pickle.load(f)
    if not isinstance(items, list):
        raise ValueError("Expected val.pkl to be a list (pouring-style dataset).")
    return items


def preprocess_video(video: torch.Tensor) -> torch.Tensor:
    # T H W C -> T C H W, [0,1]
    return video.permute(0, 3, 1, 2).float() / 255.0


def read_video_cv2(video_path: str) -> torch.Tensor:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f"No frames decoded: {video_path}")
    return torch.from_numpy(np.stack(frames, axis=0))


def center_crop_resize(video: torch.Tensor, image_size: int) -> torch.Tensor:
    # Lightweight preprocessing to match eval path more closely.
    # video shape: T C H W
    _, _, h, w = video.shape
    short = min(h, w)
    top = (h - short) // 2
    left = (w - short) // 2
    video = video[:, :, top : top + short, left : left + short]
    video = torch.nn.functional.interpolate(
        video, size=(image_size, image_size), mode="bilinear", align_corners=False
    )
    # Imagenet normalization used by most torchvision pipelines.
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=video.dtype, device=video.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=video.dtype, device=video.device).view(1, 3, 1, 1)
    return (video - mean) / std


def load_model(cfg, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    model = build_model(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model_state"]
    # Checkpoint was saved from DDP with "module." prefix.
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    msg = model.load_state_dict(state, strict=False)
    print(msg)
    model.eval()
    return model


@torch.no_grad()
def extract_embedding(model: torch.nn.Module, cfg, video_path: str, device: torch.device) -> np.ndarray:
    video = read_video_cv2(video_path)
    if video.shape[0] < 2:
        raise ValueError(f"Video too short: {video_path}")
    video = preprocess_video(video).to(device)
    video = center_crop_resize(video, cfg.IMAGE_SIZE)

    seq_len = int(video.shape[0])
    num_contexts = cfg.DATA.NUM_CONTEXTS
    steps = torch.arange(0, seq_len, dtype=torch.long, device=device)
    if num_contexts != 1:
        context_stride = cfg.DATA.CONTEXT_STRIDE
        steps = steps.view(-1, 1) + context_stride * torch.arange(
            -(num_contexts - 1), 1, dtype=torch.long, device=device
        ).view(1, -1)
        steps = torch.clamp(steps.reshape(-1), 0, seq_len - 1)

    batch = video[steps].unsqueeze(0)  # 1, T(or T*C), C, H, W
    embs = model(batch, seq_len)[0]  # T, D
    embs = torch.nn.functional.normalize(embs, dim=-1)
    return embs.detach().cpu().numpy()


def pair_sync(query_embs: np.ndarray, cand_embs: np.ndarray, stride: int) -> Tuple[List[int], float]:
    q = query_embs[::stride]
    c = cand_embs[::stride]
    d = cdist(q, c, "sqeuclidean")
    nns = np.argmin(d, axis=1)
    tau = kendalltau(np.arange(len(nns)), nns).correlation
    if np.isnan(tau):
        tau = 0.0
    return nns.tolist(), float(tau)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    cfg = load_cfg(args)
    dataset_root = cfg.PATH_TO_DATASET

    val_items = load_val_items(dataset_root)
    model = load_model(cfg, args.checkpoint, device)

    embs_map: Dict[str, np.ndarray] = {}
    for item in val_items:
        name = item["name"]
        video_path = os.path.join(dataset_root, item["video_file"])
        embs_map[name] = extract_embedding(model, cfg, video_path, device)
        print(f"[EMB] {name}: {embs_map[name].shape}")

    results = []
    names = list(embs_map.keys())
    taus = []
    for i, q_name in enumerate(names):
        for j, c_name in enumerate(names):
            if i == j:
                continue
            nns, tau = pair_sync(embs_map[q_name], embs_map[c_name], args.stride)
            taus.append(tau)
            results.append(
                {
                    "query": q_name,
                    "candidate": c_name,
                    "tau": tau,
                    "mapping": nns,
                }
            )
            print(f"[SYNC] {q_name} -> {c_name}: tau={tau:.4f}, mapping_len={len(nns)}")

    avg_tau = float(np.mean(taus)) if taus else 0.0
    print(f"\nAverage Kendall's Tau: {avg_tau:.4f}")

    if args.output_json:
        import json

        payload = {"avg_tau": avg_tau, "pairs": results}
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Saved: {args.output_json}")


if __name__ == "__main__":
    main()
