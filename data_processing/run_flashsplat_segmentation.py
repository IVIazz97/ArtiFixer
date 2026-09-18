#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Segment the Gaussians of a 3DGUT checkpoint from 2D masks, using FlashSplat.

Renders every view with its object-id mask through the FlashSplat-instrumented 3DGUT
rasterizer, accumulating each Gaussian's alpha-blending weight per label, then solves the
label assignment in closed form.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "thirdparty" / "3DGRUT-ArtiFixer", REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import torch

from flashsplat import accumulate, export, masks, solver

OUTPUT_CHOICES = ("labels", "checkpoints", "ply", "overlays")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="3DGUT checkpoint (.pt).")
    parser.add_argument("--colmap_dir", type=Path, required=True, help="COLMAP scene the checkpoint was trained on.")
    parser.add_argument(
        "--mask_dir",
        type=Path,
        required=True,
        help="Directory of single-channel object-id PNGs, one per frame, named after the frame "
        "(0 = background, 1..K = object ids).",
    )
    parser.add_argument(
        "--num_objects",
        type=int,
        default=None,
        help="Number of objects K. Defaults to the largest id found in --mask_dir.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.0,
        help="Background bias in [-1, 1]. Larger is more conservative (tighter objects).",
    )
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument(
        "--outputs",
        nargs="+",
        choices=OUTPUT_CHOICES,
        default=["labels"],
        help=f"Which artifacts to write. One or more of {OUTPUT_CHOICES}.",
    )
    parser.add_argument(
        "--contribution",
        type=Path,
        default=None,
        help="Reuse a previously saved accumulator instead of re-rendering. Lets you re-solve "
        "at a different --gamma for free.",
    )
    parser.add_argument("--downsample_factor", type=float, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    num_objects = args.num_objects
    if num_objects is None:
        num_objects = masks.infer_num_objects(args.mask_dir)
        print(f"Inferred num_objects={num_objects} from {args.mask_dir}")
    if num_objects < 1:
        raise SystemExit(f"num_objects must be >= 1, got {num_objects}")

    config_overrides = {"path": str(args.colmap_dir)}
    if args.downsample_factor is not None:
        config_overrides["dataset.downsample_factor"] = args.downsample_factor

    model, conf, global_step = accumulate.load_model(args.checkpoint, config_overrides)
    dataset, dataloader = accumulate.build_test_dataloader(conf)
    print(f"Loaded {model.num_gaussians} Gaussians, {len(dataset)} views")

    args.output_root.mkdir(parents=True, exist_ok=True)

    if args.contribution is not None:
        contribution = torch.load(args.contribution).cuda()
        if contribution.shape != (num_objects + 1, model.num_gaussians):
            raise SystemExit(
                f"{args.contribution} has shape {tuple(contribution.shape)}, expected "
                f"{(num_objects + 1, model.num_gaussians)}"
            )
        print(f"Reusing accumulator from {args.contribution}")
    else:
        contribution = accumulate.accumulate_contributions(
            model, dataset, dataloader, args.mask_dir, num_objects
        )

    labels = solver.multi_instance_opt(contribution, gamma=args.gamma)
    counts = {int(i): int(labels[i].sum()) for i in range(labels.shape[0])}
    print(f"Gaussians per label (0 = background): {counts}")

    if "labels" in args.outputs:
        torch.save(contribution.cpu(), args.output_root / "contribution.pt")
        torch.save(labels.cpu(), args.output_root / "labels.pt")
        (args.output_root / "summary.json").write_text(
            json.dumps(
                {
                    "num_objects": num_objects,
                    "num_gaussians": int(model.num_gaussians),
                    "gamma": args.gamma,
                    "global_step": global_step,
                    "gaussians_per_label": counts,
                },
                indent=2,
            )
        )

    for object_id in range(1, num_objects + 1):
        keep = labels[object_id]
        if not bool(keep.any()):
            print(f"object {object_id}: no Gaussians assigned, skipping export")
            continue

        if "checkpoints" in args.outputs:
            kept = export.save_segmented_checkpoint(
                model, keep, args.output_root / f"object_{object_id:03d}" / "ckpt.pt", global_step
            )
            removed = export.save_segmented_checkpoint(
                model, ~keep, args.output_root / f"object_{object_id:03d}" / "ckpt_removed.pt", global_step
            )
            print(f"object {object_id}: checkpoints {kept} kept / {removed} complement")

        if "ply" in args.outputs:
            export.save_segmented_ply(model, keep, args.output_root / f"object_{object_id:03d}" / "object.ply")
            export.save_segmented_ply(
                model, ~keep, args.output_root / f"object_{object_id:03d}" / "removed.ply"
            )

    if "overlays" in args.outputs:
        written = export.save_overlay_renders(
            model, dataset, dataloader, labels, args.output_root / "overlays"
        )
        print(f"Wrote {written} overlay renders")

    print(f"Done. Outputs in {args.output_root}")


if __name__ == "__main__":
    main()
