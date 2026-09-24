#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inpaint360GS stage 5 (ArtiFixer env): virtual camera views around the removed object (Sec. 3.4).

As the official ``tools/virtual_pose.py``: 30 poses on a PCA-aligned circle around the capture's
focus point, at the radius where the removed object fills ~70% of the view
(``inpaint360gs.poses``), with the intrinsics of the first view (by image name). Each pose is
rendered from the full scene and from the scene with the target and surrounding objects
removed: RGB, camera z-depth and opacity. The removed Gaussians' visible share of the full
render (sum of alpha * T over them, so Gaussians hidden inside or under other surfaces do not
count) is the object's footprint, the automatic stand-in for the paper's interactive SAM-Track
selection (``nbs_masks``).

Depth is the alpha-weighted mean z (3DGUT pred_dist / opacity), not the official un-normalised
sum of z * alpha * T; the two agree wherever the render is opaque.

Writes <output_dir>/virtual/{cameras.json, virtual.json, full/{rgb,depth}, removed/{rgb,depth,
opacity}, footprint/, target_removed/rgb (only with surrounding ids)}/<i>.png|.npy.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from flashsplat import export
from inpaint360gs.common import (
    ViewCamera, all_views_overrides, build_dataset, dataset_views, load_model, render_features, render_view,
    save_cameras, save_png, virtual_batch,
)
from inpaint360gs.poses import circle_path, virtual_radius


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--colmap_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True, help="Holds removal/ from remove.py.")
    parser.add_argument("--n_frames", type=int, default=30)
    parser.add_argument("--circle_radius", type=float, default=-1.0,
                        help="Ratio of the camera spread; smaller is closer. Default: from the object radius.")
    args = parser.parse_args()

    root = args.output_dir
    out = root / "virtual"
    removal = json.loads((root / "removal" / "removal.json").read_text())
    masks = torch.load(root / "removal" / "removal.pt", map_location="cuda")
    model, conf, _ = load_model(args.checkpoint, all_views_overrides(args.colmap_dir))
    dataset, loader = build_dataset(conf)
    batches, cameras = dataset_views(dataset, loader)

    first = min(range(len(cameras)), key=lambda i: cameras[i].name)
    template, reference = batches[first], cameras[first]
    c2ws = np.stack([c.c2w for c in cameras])
    fov_x = 2 * math.atan(reference.width / (2 * reference.fx))
    fov_y = 2 * math.atan(reference.height / (2 * reference.fy))
    circle_radius = args.circle_radius
    if circle_radius <= 0:
        circle_radius = virtual_radius(c2ws, fov_x, fov_y, removal["target_object_radius"])
    poses = circle_path(c2ws, args.n_frames, circle_radius)
    print(f"{args.n_frames} virtual views, circle_radius {circle_radius:.4f}")

    removed = masks["removed"]
    models = {"full": model, "removed": export.segment_model(model, ~removed, setup_optimizer=False)}
    footprint = masks["footprint"].float()[:, None]
    if removal["surrounding_ids"]:
        models["target_removed"] = export.segment_model(model, ~removed | masks["surrounding"], setup_optimizer=False)
    for m in models.values():
        m.renderer = model.renderer

    virtual_cameras = []
    for i, c2w in enumerate(poses):
        batch = virtual_batch(template, c2w)
        virtual_cameras.append(ViewCamera(index=i, name=f"{i:05d}", width=reference.width, height=reference.height,
                                          fx=reference.fx, fy=reference.fy, cx=reference.cx, cy=reference.cy,
                                          c2w=c2w))
        with torch.no_grad():
            save_png(render_features(model, batch, footprint)[..., 0].clamp(0, 1), out / "footprint" / f"{i:05d}.png")
        for name, m in models.items():
            rendered = render_view(m, batch, 0)
            save_png(rendered["rgb"], out / name / "rgb" / f"{i:05d}.png")
            if name in ("full", "removed"):
                (out / name / "depth").mkdir(parents=True, exist_ok=True)
                np.save(out / name / "depth" / f"{i:05d}.npy", rendered["depth"].cpu().numpy().astype(np.float32))
            if name == "removed":
                save_png(rendered["opacity"], out / name / "opacity" / f"{i:05d}.png")
    save_cameras(virtual_cameras, out / "cameras.json")
    (out / "virtual.json").write_text(json.dumps({
        "circle_radius": circle_radius, "n_frames": args.n_frames, "reference_view": reference.name,
        "target_object_radius": removal["target_object_radius"],
    }, indent=1))
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
