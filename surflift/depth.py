#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SurfaceLift stage 5 (artifixer env): MoGe-2 ViT-L metric depth of the inpainted view, given the
camera's true horizontal field of view (SurfaceLift scripts/depth_view.py). Only scale and shift
remain for surflift.align to fit against the rendered depth.
-> select/mono_depth.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from surflift.common import Timer, fov_x, job_summary, load_view


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--moge", type=Path, required=True, help="moge-2-vitl-normal/model.pt")
    args = ap.parse_args()
    d = args.work / "select"
    timer = Timer(args.work, "depth")
    view = load_view(d / "camera.json")
    from moge.model.v2 import MoGeModel
    model = MoGeModel.from_pretrained(str(args.moge)).cuda().eval()
    img = torch.tensor(np.asarray(Image.open(d / "inpainted.png").convert("RGB")), dtype=torch.float32, device="cuda").permute(2, 0, 1) / 255
    with torch.no_grad():
        out = model.infer(img, fov_x=fov_x(view))
    depth = out["depth"].cpu().numpy().astype(np.float32)
    depth[~np.isfinite(depth)] = 0
    if not (depth > 0).mean() > 0.5:
        raise SystemExit(f"depth: MoGe returned valid depth on only {100 * (depth > 0).mean():.1f} % of {d / 'inpainted.png'}")
    np.save(d / "mono_depth.npy", depth)
    job_summary(args.work, f"depth: MoGe-2 median {np.median(depth[depth > 0]):.3f} (fov_x {fov_x(view):.1f} deg)")
    timer.done(median_depth=float(np.median(depth[depth > 0])) if (depth > 0).any() else 0.0)


if __name__ == "__main__":
    main()
