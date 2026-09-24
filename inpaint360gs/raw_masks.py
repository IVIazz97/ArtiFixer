#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inpaint360GS stage 1 (aurafusion env): per-view class-agnostic 2D masks.

As the official ``seg/raw_mask_sam.py``: every photo is segmented independently and written as
an id map (0 = unsegmented, 1..M = masks of this view only; ids are not consistent across views
until ``associate``). The official default is CropFormer entity segmentation ("hqsam"); this
uses SAM2 hiera-large automatic masks with the official ``sam`` settings (64 points per side,
predicted-IoU threshold 0.8). Objects SAM2 will not propose at those thresholds need looser ones:
on bonsai the tree is missed entirely, and ``--pred_iou_thresh 0.7 --stability_score_thresh 0.85``
finds it (0.6% -> 75% of its reference-mask pixels covered). SAM masks nest (part inside whole), so they are painted largest
first and every pixel keeps the smallest mask that covers it.

Writes <output_dir>/raw_masks/<image stem>.png (16-bit).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def paint_masks(records: list[dict], shape: tuple[int, int], min_area: int) -> np.ndarray:
    """Id map from SAM records, largest mask first so nested smaller masks stay visible."""
    ids = np.zeros(shape, dtype=np.uint16)
    kept = [r for r in records if r["area"] >= min_area]
    for label, record in enumerate(sorted(kept, key=lambda r: -r["area"]), start=1):
        ids[record["segmentation"]] = label
    # Masks fully covered by smaller ones leave gaps in the numbering; compact it.
    present = np.unique(ids)
    present = present[present > 0]
    lut = np.zeros(int(ids.max()) + 1, dtype=np.uint16)
    lut[present] = np.arange(1, len(present) + 1, dtype=np.uint16)
    return lut[ids]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image_dir", type=Path, required=True, help="The COLMAP scene's images/ directory.")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--sam2_model", default="facebook/sam2-hiera-large")
    parser.add_argument("--points_per_side", type=int, default=64)
    parser.add_argument("--points_per_batch", type=int, default=64)
    parser.add_argument("--pred_iou_thresh", type=float, default=0.8)
    parser.add_argument("--stability_score_thresh", type=float, default=0.95, help="SAM2's default.")
    parser.add_argument("--crop_n_layers", type=int, default=0, help="1 also segments image crops.")
    parser.add_argument("--min_area", type=int, default=0, help="Drop masks smaller than this many pixels.")
    parser.add_argument("--overwrite", action="store_true", help="Redo views whose mask already exists.")
    args = parser.parse_args()

    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    generator = SAM2AutomaticMaskGenerator.from_pretrained(
        args.sam2_model, points_per_side=args.points_per_side, points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_thresh, stability_score_thresh=args.stability_score_thresh,
        crop_n_layers=args.crop_n_layers,
    )
    images = sorted(p for p in args.image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    out = args.output_dir / "raw_masks"
    out.mkdir(parents=True, exist_ok=True)
    counts = []
    for i, path in enumerate(images):
        if (out / f"{path.stem}.png").exists() and not args.overwrite:
            counts.append(int(cv2.imread(str(out / f"{path.stem}.png"), cv2.IMREAD_UNCHANGED).max()))
            continue
        image = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            records = generator.generate(image)
        ids = paint_masks(records, image.shape[:2], args.min_area)
        cv2.imwrite(str(out / f"{path.stem}.png"), ids)
        counts.append(int(ids.max()))
        if i % 25 == 0:
            print(f"{i + 1}/{len(images)} {path.name}: {counts[-1]} masks", flush=True)
    print(f"Done: {len(images)} views, masks per view mean={np.mean(counts):.1f} max={max(counts)} -> {out}")


if __name__ == "__main__":
    main()
