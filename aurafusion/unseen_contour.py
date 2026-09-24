#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AuraFusion stage 2: depth-aware unseen contour (paper Eq. 1-2).

For view n, every pixel of its removal region R_n is lifted to 3D with the removed-scene depth
and projected into every other view i in which the object is visible. The pixel counts as
hidden in view i if it lands inside R_i. C_n = [mean_i hidden > theta] & R_n, theta = 0.6.

Same as the official ``GaussianExtractor.depth_aware_unseen_mask_generation`` with
``removal_region=depth_diff``: R is where the object changes the render (here: object opacity
above ``--region_threshold``), no dilation; projections outside the image count as seen.
One addition: points behind a camera count as seen (the official warp does not test z).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from aurafusion.scene import load_cameras, project, save_png, unproject


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render_dir", type=Path, required=True, help="render_views.py output.")
    parser.add_argument("--region_threshold", type=float, default=0.02, help="Object opacity defining R_i.")
    parser.add_argument("--aggregate_threshold", type=float, default=0.6, help="theta in Eq. 2.")
    parser.add_argument("--chunk", type=int, default=65536)
    args = parser.parse_args()

    root = args.render_dir
    cameras = load_cameras(root / "cameras.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    regions = torch.stack([
        torch.from_numpy(np.asarray(Image.open(root / "object" / "opacity" / f"{c.index:05d}.png"), dtype=np.float32) / 255.0)
        > args.region_threshold
        for c in cameras
    ]).to(device)  # [J, H, W]
    J, H, W = regions.shape
    has_object = regions.flatten(1).any(1)

    fractions = []
    for n, camera in enumerate(cameras):
        region = regions[n]
        contour = torch.zeros_like(region)
        if region.any():
            depth = torch.from_numpy(np.load(root / "removed" / "depth" / f"{n:05d}.npy")).to(device)
            points = unproject(camera, depth, region)
            others = [j for j in range(J) if j != n and bool(has_object[j])]
            hidden = torch.zeros(points.shape[0], device=device)
            for s in range(0, points.shape[0], args.chunk):
                u, v, z = project([cameras[j] for j in others], points[s:s + args.chunk])
                inside = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0)
                flat = torch.as_tensor(others, device=device)[:, None] * (H * W) + v.clamp(0, H - 1) * W + u.clamp(0, W - 1)
                hidden[s:s + args.chunk] = (regions.flatten()[flat] & inside).float().sum(0)
            fraction = hidden / max(len(others), 1)
            contour[region] = fraction > args.aggregate_threshold
            fractions.append(float(contour.sum()) / float(region.sum()))
        save_png(contour, root / "unseen_contour" / f"{n:05d}.png")
        if n % 50 == 0:
            print(f"view {n}: |R|={int(region.sum())} |C|={int(contour.sum())}", flush=True)
    print(f"Done: {J} views, contour/region area ratio mean={np.mean(fractions):.3f} "
          f"min={np.min(fractions):.3f} max={np.max(fractions):.3f}")


if __name__ == "__main__":
    main()
