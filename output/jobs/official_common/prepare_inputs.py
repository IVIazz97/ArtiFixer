#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared inputs for the official video-prior baselines (Omni-3DEdit, MVInpainter) on a mip360 scene.

Both official repos take plain images (no poses, no 3D model), and both need a per-view object
mask to be put back into a full-resolution frame for the 3DGS refit, so the masks are built once
here, with the same footprint our ArtiFixer removal variants use (``build_removal_split.py``):
SAM2 mask union projected FlashSplat object opacity > 0.05, dilated by 12 px. For garden the
FlashSplat dir is ``garden_vase`` (table + the vase standing on it), as in ArtiFixer v5.

    <out>/images/<stem>.png     the photo our 3DGUT reconstruction was trained on
    <out>/masks/<stem>.png      dilated object footprint, 255 = object
    <out>/views.json            name-sorted stems, reference stem and mask stats
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--colmap_dir", type=Path, required=True, help="$RECON/<scene>/3dgrut_input/<scene>.")
    parser.add_argument("--flashsplat_dir", type=Path, required=True)
    parser.add_argument("--mask_dir", type=Path, required=True, help="datasets/masks_objid/<scene> (SAM2 .npy).")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--reference", required=True, help="Stem of the view that gets the 2D edit.")
    parser.add_argument("--dilate_px", type=int, default=12)
    parser.add_argument("--fg_opacity_threshold", type=float, default=0.05)
    args = parser.parse_args()

    index_by_name = {n: int(i) for i, n in json.loads((args.flashsplat_dir / "frame_names.json").read_text()).items()}
    (args.output_dir / "images").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "masks").mkdir(parents=True, exist_ok=True)
    stems, fractions = [], []
    for photo_path in sorted((args.colmap_dir / "images").iterdir()):
        stem = photo_path.stem
        photo = Image.open(photo_path).convert("RGB")
        w, h = photo.size
        sam = np.load(args.mask_dir / f"{stem}.npy")
        if sam.shape != (h, w):
            sam = np.asarray(Image.fromarray(sam).resize((w, h), Image.NEAREST))
        fg = np.asarray(Image.open(args.flashsplat_dir / "foreground_renders_opacity"
                                   / f"{index_by_name[photo_path.name]:05d}.png").convert("L"))
        assert fg.shape == (h, w), f"{stem}: opacity {fg.shape} vs photo {(h, w)}"
        mask = binary_dilation((sam > 0) | (fg / 255.0 > args.fg_opacity_threshold), iterations=args.dilate_px)
        photo.save(args.output_dir / "images" / f"{stem}.png")
        Image.fromarray((mask * 255).astype(np.uint8)).save(args.output_dir / "masks" / f"{stem}.png")
        stems.append(stem)
        fractions.append(float(mask.mean()))

    assert args.reference in stems, f"reference {args.reference} not among the views"
    (args.output_dir / "views.json").write_text(json.dumps({
        "stems": stems, "reference": args.reference, "width": w, "height": h,
        "dilate_px": args.dilate_px, "mask_fraction": {"mean": float(np.mean(fractions)), "max": float(np.max(fractions))},
    }, indent=1))
    print(f"{len(stems)} views ({w}x{h}) -> {args.output_dir}; mask fraction mean {np.mean(fractions):.3f} "
          f"max {np.max(fractions):.3f}; reference {args.reference}")


if __name__ == "__main__":
    main()
