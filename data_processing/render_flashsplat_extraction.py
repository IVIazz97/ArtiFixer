#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render foreground-only and background-only 3DGUT views from FlashSplat labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "thirdparty" / "3DGRUT-ArtiFixer", REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import torch
import torchvision
import numpy as np
from scipy.spatial import ConvexHull, QhullError

from flashsplat import accumulate, export


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--colmap_dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument(
        "--contribution",
        type=Path,
        default=None,
        help="Saved [num_objects + 1, num_gaussians] contribution tensor for global filtering.",
    )
    parser.add_argument(
        "--min-total-contribution",
        type=float,
        default=None,
        help="Keep a Gaussian in either output only when total accumulated contribution is above this value.",
    )
    parser.add_argument(
        "--hit-count",
        type=Path,
        default=None,
        help="Saved per-Gaussian count of views with positive contribution.",
    )
    parser.add_argument(
        "--normalize-by-hit-count",
        action="store_true",
        help="Use total contribution divided by per-view hit count for thresholding.",
    )
    parser.add_argument(
        "--background-convex-hull",
        action="store_true",
        help="Remove background Gaussians whose centers lie inside the 3D convex hull of "
        "the retained foreground Gaussian centers.",
    )
    parser.add_argument(
        "--convex-hull-trim-percentile",
        type=float,
        default=100.0,
        help="Before building the foreground hull, keep centers within this percentile "
        "of centroid distance. 100 keeps the untrimmed hull.",
    )
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--object_id", type=int, default=1)
    parser.add_argument(
        "--background",
        choices=("complement", "label0"),
        default="complement",
        help="Use complement of object_id by default; label0 uses the independently solved background row.",
    )
    parser.add_argument("--downsample_factor", type=float, default=None)
    parser.add_argument("--test_split_interval", type=int, default=None)
    parser.add_argument("--foreground_dirname", default="foreground_renders")
    parser.add_argument("--background_dirname", default="background_renders")
    parser.add_argument(
        "--view_stride",
        type=int,
        default=1,
        help="Render every Nth dataset view; output filenames retain dataset indices.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.view_stride < 1:
        raise SystemExit("--view_stride must be >= 1")
        if not 0 < args.convex_hull_trim_percentile <= 100:
            raise SystemExit("--convex-hull-trim-percentile must be in (0, 100]")
    labels = torch.load(args.labels, map_location="cpu")
    if labels.ndim != 2:
        raise SystemExit(f"Expected labels [K+1,N], got {tuple(labels.shape)}")
    if args.object_id < 1 or args.object_id >= labels.shape[0]:
        raise SystemExit(f"object_id must be in [1, {labels.shape[0] - 1}], got {args.object_id}")
    if args.min_total_contribution is not None and args.min_total_contribution < 0:
        raise SystemExit("--min-total-contribution must be >= 0")
    if args.min_total_contribution is not None and args.contribution is None:
        raise SystemExit("--contribution is required with --min-total-contribution")
    if args.normalize_by_hit_count and args.hit_count is None:
        raise SystemExit("--hit-count is required with --normalize-by-hit-count")

    config_overrides = {"path": str(args.colmap_dir)}
    if args.downsample_factor is not None:
        config_overrides["dataset.downsample_factor"] = args.downsample_factor
    if args.test_split_interval is not None:
        config_overrides["dataset.test_split_interval"] = args.test_split_interval

    model, conf, global_step = accumulate.load_model(args.checkpoint, config_overrides)
    dataset, dataloader = accumulate.build_test_dataloader(conf)

    foreground_keep = labels[args.object_id].to(model.positions.device)
    if args.contribution is not None:
        contribution = torch.load(args.contribution, map_location="cpu")
        expected_shape = (labels.shape[0], model.num_gaussians)
        if tuple(contribution.shape) != expected_shape:
            raise SystemExit(
                f"{args.contribution} has shape {tuple(contribution.shape)}, expected {expected_shape}"
            )
        if args.min_total_contribution is not None:
            total_contribution = contribution.sum(dim=0)
            if args.normalize_by_hit_count:
                hit_count = torch.load(args.hit_count, map_location="cpu").float()
                if tuple(hit_count.shape) != (model.num_gaussians,):
                    raise SystemExit(
                        f"{args.hit_count} has shape {tuple(hit_count.shape)}, expected "
                        f"({model.num_gaussians},)"
                    )
                total_contribution = total_contribution / hit_count.clamp_min(1.0)
            visible = (total_contribution > args.min_total_contribution).to(
                model.positions.device
            )
            foreground_keep &= visible
        else:
            visible = None
    else:
        visible = None
    if args.background == "label0":
        background_keep = labels[0].to(model.positions.device)
    else:
        background_keep = ~foreground_keep
    if visible is not None:
        background_keep &= visible.to(background_keep.device)
    if args.background_convex_hull:
        foreground_positions = model.positions.detach()[foreground_keep].cpu().numpy()
        if args.convex_hull_trim_percentile < 100:
            centroid = foreground_positions.mean(axis=0)
            distances = np.linalg.norm(foreground_positions - centroid, axis=1)
            cutoff = np.percentile(distances, args.convex_hull_trim_percentile)
            foreground_positions = foreground_positions[distances <= cutoff]
        if foreground_positions.shape[0] < 4:
            raise SystemExit("At least four foreground Gaussians are required for a 3D convex hull")
        if foreground_positions.shape[0] > 10_000:
            rng = np.random.default_rng(0)
            selected = rng.choice(foreground_positions.shape[0], 10_000, replace=False)
            foreground_positions = foreground_positions[selected]
        try:
            hull = ConvexHull(foreground_positions)
        except QhullError as error:
            raise SystemExit(f"Could not compute foreground 3D convex hull: {error}") from error

        background_indices = torch.where(background_keep)[0]
        background_indices_cpu = background_indices.cpu()
        background_positions = model.positions.detach()[background_indices].cpu().numpy()
        hull_equations = hull.equations[:, :3]
        hull_offsets = hull.equations[:, 3]
        scale = max(float((foreground_positions.max(axis=0) - foreground_positions.min(axis=0)).max()), 1.0)
        tolerance = 1e-7 * scale
        inside = torch.empty(background_indices.numel(), dtype=torch.bool)
        for start in range(0, background_indices.numel(), 65536):
            stop = min(start + 65536, background_indices.numel())
            points = background_positions[start:stop]
            inequalities = points @ hull_equations.T + hull_offsets
            inside[start:stop] = torch.from_numpy((inequalities <= tolerance).all(axis=1))
        background_keep[background_indices_cpu[inside].to(background_keep.device)] = False

    print(
        f"Loaded step={global_step}, views={len(dataset)}, gaussians={model.num_gaussians}; "
        f"foreground={int(foreground_keep.sum())}, background={int(background_keep.sum())}"
        f"{', min total contribution > ' + str(args.min_total_contribution) if args.min_total_contribution is not None else ''}"
        f"{', convex-hull background filter' if args.background_convex_hull else ''}",
        f"{', hull trim percentile=' + str(args.convex_hull_trim_percentile) if args.background_convex_hull else ''}",
        flush=True,
    )

    foreground_model = export.segment_model(model, foreground_keep, setup_optimizer=False)
    foreground_model.renderer = model.renderer
    background_model = export.segment_model(model, background_keep, setup_optimizer=False)
    background_model.renderer = model.renderer

    foreground_dir = args.output_root / args.foreground_dirname
    background_dir = args.output_root / args.background_dirname
    foreground_dir.mkdir(parents=True, exist_ok=True)
    background_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            if index % args.view_stride != 0:
                continue
            gpu_batch = dataset.get_gpu_batch_with_intrinsics(batch)
            foreground = foreground_model(gpu_batch, train=False, frame_id=index)["pred_rgb"][0].clamp(0, 1)
            background = background_model(gpu_batch, train=False, frame_id=index)["pred_rgb"][0].clamp(0, 1)
            torchvision.utils.save_image(foreground.permute(2, 0, 1), foreground_dir / f"{index:05d}.png")
            torchvision.utils.save_image(background.permute(2, 0, 1), background_dir / f"{index:05d}.png")
            written += 1
            if written == 1 or written % 20 == 0 or written == len(dataset):
                print(f"rendered {written}/{len(dataset)} views", flush=True)

    print(f"Done. Wrote {written} foreground renders to {foreground_dir}")
    print(f"Done. Wrote {written} background renders to {background_dir}")


if __name__ == "__main__":
    main()
