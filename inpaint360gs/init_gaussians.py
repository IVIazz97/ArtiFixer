#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inpaint360GS stage 8 (ArtiFixer env): depth-guided Gaussian initialisation (Sec. 3.5).

As the official ``edit_object_removal_plyfusion.py`` + ``GaussianModel.inpaint_setup``: the
masked pixels of ONE virtual view (``--seed_view``, 4 as hard-coded in the official
``edit_object_inpaint.py``) are unprojected with the LaMa-completed depth and coloured with the
LaMa-inpainted image; Open3D-style statistical outlier removal (5 neighbours, std ratio 4.0)
cleans the cloud. Every point becomes a Gaussian whose colour (SH DC) and position come from the
cloud, whose opacity is 0.1, and whose scale, rotation and higher SH bands are the mean of its 5
nearest remaining-scene Gaussians (the official ``smarter_ini``).

The remaining scene (target and surrounding objects removed) comes first in the checkpoint;
``inpaint360gs_num_fixed`` records how many Gaussians it has.

Writes <output_dir>/init/{ckpt_init.pt, virtual_rgb/<i>.png}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree

from inpaint360gs.common import (
    SH_C0, all_views_overrides, build_dataset, load_cameras, load_model, render_view, save_png, template_batch, unproject,
    virtual_batch,
)


def statistical_outliers(points: np.ndarray, nb_neighbors: int = 5, std_ratio: float = 4.0) -> np.ndarray:
    """Open3D ``remove_statistical_outlier``: keep points whose mean distance to their
    ``nb_neighbors`` nearest (self included) is below mean + std_ratio * std of that statistic."""
    if len(points) <= nb_neighbors:
        return np.ones(len(points), dtype=bool)
    distances, _ = cKDTree(points).query(points, k=nb_neighbors)
    mean = distances.mean(axis=1)
    return mean <= mean.mean() + std_ratio * mean.std(ddof=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True, help="Holds removal/, virtual/, nbs/ and lama/.")
    parser.add_argument("--colmap_dir", type=Path, required=True)
    parser.add_argument("--seed_view", type=int, default=4)
    parser.add_argument("--opacity_init", type=float, default=0.1)
    parser.add_argument("--knn", type=int, default=5)
    args = parser.parse_args()

    root, s = args.output_dir, args.seed_view
    out = root / "init"
    cameras = load_cameras(root / "virtual" / "cameras.json")
    camera = cameras[s]
    mask = torch.from_numpy(np.asarray(Image.open(root / "nbs" / "mask" / f"{s:05d}.png").convert("L")) > 127).cuda()
    depth = torch.from_numpy(np.load(root / "lama" / "depth" / f"{s:05d}.npy")).cuda()
    color = torch.from_numpy(np.asarray(Image.open(root / "lama" / "rgb" / f"{s:05d}.png").convert("RGB"))).cuda()
    points = unproject(camera, depth, mask)
    colors = color[mask].float() / 255.0
    keep = torch.from_numpy(statistical_outliers(points.cpu().numpy().astype(np.float64))).cuda()
    points, colors = points[keep], colors[keep]
    print(f"seed view {s}: {int(mask.sum())} hole pixels, {points.shape[0]} points after outlier removal")

    model, conf, checkpoint = load_model(root / "removal" / "ckpt_removed.pt", all_views_overrides(args.colmap_dir))
    num_fixed = model.num_gaussians
    _, neighbours = cKDTree(model.positions.detach().cpu().numpy()).query(points.cpu().numpy(), k=args.knn)
    neighbours = torch.from_numpy(neighbours).cuda()
    mean_of = lambda t: t.detach()[neighbours].mean(dim=1)  # noqa: E731
    n = points.shape[0]
    new = {
        "positions": points,
        "rotation": mean_of(model.rotation),
        "scale": mean_of(model.scale),
        "density": model.density_activation_inv(torch.full((n, 1), args.opacity_init, device="cuda")),
        "features_albedo": (colors - 0.5) / SH_C0,
        "features_specular": mean_of(model.features_specular),
    }
    for name, value in new.items():
        old = getattr(model, name)
        setattr(model, name, torch.nn.Parameter(torch.cat([old.detach(), value.to(old.dtype)], dim=0)))
    model.set_optimizable_parameters()
    model.setup_optimizer()
    model.validate_fields()
    model.build_acc()
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.get_model_parameters() | {"global_step": int(checkpoint["global_step"]), "epoch": 0,
                                               "inpaint360gs_num_fixed": num_fixed}, out / "ckpt_init.pt")
    print(f"checkpoint: {num_fixed} fixed + {n} new Gaussians -> {out / 'ckpt_init.pt'}")

    dataset, _ = build_dataset(conf, num_workers=0)
    template = template_batch(dataset, json.loads((root / "virtual" / "virtual.json").read_text())["reference_view"])
    for camera in cameras:
        rendered = render_view(model, virtual_batch(template, camera.c2w), 0)
        save_png(rendered["rgb"], out / "virtual_rgb" / f"{camera.index:05d}.png")
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
