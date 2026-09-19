#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compute PSNR, SSIM, and LPIPS for local COLMAP-style 3DGUT renders."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_eval.metrics_utils import MetricsAggregator, compute_rgb_metrics, load_image_as_tensor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--render_dir", type=Path, required=True, help="Directory containing 00000.png-style renders.")
    parser.add_argument("--gt_image_dir", type=Path, required=True, help="Directory containing source/ground-truth images.")
    parser.add_argument(
        "--transforms_json",
        type=Path,
        required=True,
        help="NeRFStudio transforms.json whose frame order matches render indices.",
    )
    parser.add_argument(
        "--selected_indices",
        type=Path,
        required=True,
        help="JSON list of training frame indices. Held-out metrics use frames not in this list.",
    )
    parser.add_argument("--output_yaml", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lpips_net_type", default="vgg")
    parser.add_argument(
        "--split",
        choices=("heldout", "all"),
        default="heldout",
        help="Evaluate held-out frames by default; use all for reconstruction sanity checks.",
    )
    parser.add_argument(
        "--include_per_frame",
        action="store_true",
        help="Include per-frame metric values in the output YAML.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests. Full evaluation uses all selected frames by default.",
    )
    return parser


def load_frame_paths(transforms_json: Path) -> list[str]:
    with transforms_json.open() as file:
        transforms = json.load(file)
    frames = transforms.get("frames")
    if not isinstance(frames, list):
        raise ValueError(f"{transforms_json} has no frames list")
    paths = []
    for frame in frames:
        file_path = frame.get("file_path")
        if not isinstance(file_path, str):
            raise ValueError(f"Invalid frame entry in {transforms_json}: {frame!r}")
        paths.append(file_path)
    return paths


def load_training_indices(path: Path) -> set[int]:
    indices = json.loads(path.read_text())
    if not isinstance(indices, list) or not all(isinstance(index, int) for index in indices):
        raise ValueError(f"{path} must contain a JSON list of integer frame indices")
    return set(indices)


def resolve_gt_path(gt_image_dir: Path, frame_file_path: str) -> Path:
    basename = Path(frame_file_path).name
    candidates = [gt_image_dir / frame_file_path, gt_image_dir / basename]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing GT image for {frame_file_path!r} under {gt_image_dir}")


def assert_same_size(render_path: Path, gt_path: Path) -> None:
    with Image.open(render_path) as render_image, Image.open(gt_path) as gt_image:
        if render_image.size != gt_image.size:
            raise ValueError(
                f"Image size mismatch for {render_path.name}: render={render_image.size}, gt={gt_image.size}"
            )


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    frame_paths = load_frame_paths(args.transforms_json)
    training_indices = load_training_indices(args.selected_indices)
    if args.split == "heldout":
        frame_indices = [index for index in range(len(frame_paths)) if index not in training_indices]
    else:
        frame_indices = list(range(len(frame_paths)))
    if args.max_frames is not None:
        if args.max_frames < 1:
            raise SystemExit(f"--max_frames must be >= 1, got {args.max_frames}")
        frame_indices = frame_indices[: args.max_frames]
    if not frame_indices:
        raise SystemExit(f"No frames selected for split={args.split}")

    aggregator = MetricsAggregator(["psnr", "ssim", "lpips"])
    compared = 0
    with torch.inference_mode():
        for frame_idx in frame_indices:
            render_path = args.render_dir / f"{frame_idx:05d}.png"
            if not render_path.is_file():
                raise FileNotFoundError(f"Missing render: {render_path}")
            gt_path = resolve_gt_path(args.gt_image_dir, frame_paths[frame_idx])
            assert_same_size(render_path, gt_path)

            render = load_image_as_tensor(render_path, device)
            target = load_image_as_tensor(gt_path, device)
            aggregator.add(args.scene_id, **compute_rgb_metrics(render, target, args.lpips_net_type))
            compared += 1
            if compared == 1 or compared % 20 == 0 or compared == len(frame_indices):
                print(f"{args.scene_id}: compared {compared}/{len(frame_indices)} frames", flush=True)

    args.output_yaml.parent.mkdir(parents=True, exist_ok=True)
    aggregator.print_scene_summary(args.scene_id)
    aggregator.save_to_yaml(
        args.output_yaml,
        include_per_frame=args.include_per_frame,
        frame_count=compared,
        train_frame_count=len(training_indices),
        total_frame_count=len(frame_paths),
    )


if __name__ == "__main__":
    main()
